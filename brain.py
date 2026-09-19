import json
import math
import os
import statistics
import sys
from dataclasses import dataclass

import optuna
from optuna.distributions import CategoricalDistribution, IntDistribution
from optuna.trial import TrialState

from apply_gpu_settings import FatalError, InvalidConfig, UnstableConfig, apply_gpu_settings
from run_cyberpunk2077 import run_cyberpunk2077

# ==========================================
# --- CONSTANTS ---
# ==========================================
CONFIG_FILE = "config.json"
STATE_FILE = "state.json"
DB_URL = "sqlite:///overclock_study.db"
STUDY_NAME = "gpu_tuning_session"
N_TRIALS = 50

VERIFY_WITHIN = 0.02        # re-run the benchmark if the scout score is within 2% of the best
VERIFY_RUNS = 2             # extra runs for verification (median of scout + these)
MAX_BENCH_FAILURES_IN_A_ROW = 5  # after this many, assume the harness is broken, not the settings


class ConfigError(Exception):
    """config.json is missing, malformed, or inconsistent."""


class BenchmarkFailed(Exception):
    """The benchmark crashed or produced a nonsense score (treated as instability)."""


# ==========================================
# --- MODULE 1: CONFIG ---
# ==========================================
@dataclass(frozen=True)
class Settings:
    crash_penalty: float
    voltage_penalty_weight: float
    space: dict  # name -> optuna distribution; the single source of truth for the search space


def _require(cfg, *keys):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise ConfigError(f"{CONFIG_FILE} is missing '{'.'.join(keys)}'")
        node = node[k]
    return node


def _number(cfg, *keys):
    v = _require(cfg, *keys)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ConfigError(f"'{'.'.join(keys)}' must be a finite number, got {v!r}")
    return v


def _int_range(cfg, lo_key, hi_key, step):
    lo = _number(cfg, "params", lo_key)
    hi = _number(cfg, "params", hi_key)
    if not (isinstance(lo, int) and isinstance(hi, int)):
        raise ConfigError(f"'params.{lo_key}' and 'params.{hi_key}' must be integers")
    if lo > hi:
        raise ConfigError(f"'params.{lo_key}' ({lo}) is greater than 'params.{hi_key}' ({hi})")
    return IntDistribution(lo, hi, step=step)


def load_config(path=CONFIG_FILE):
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise ConfigError(f"{path} not found")
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path} is not valid JSON: {e}")

    return Settings(
        crash_penalty=_number(cfg, "tuning", "crash_penalty"),
        voltage_penalty_weight=_number(cfg, "tuning", "voltage_penalty_weight"),
        space={
            "freq_gpu": _int_range(cfg, "gpu_freq_min", "gpu_freq_max", step=10),
            "memfreq": _int_range(cfg, "memory_freq_min", "memory_freq_max", step=10),
            "mem_timing": CategoricalDistribution([0, 1]),
            "power_limit": _int_range(cfg, "power_percentage_offset_min", "power_percentage_offset_max", step=1),
            "voltage": _int_range(cfg, "voltage_offset_min", "voltage_offset_max", step=5),
        },
    )


def suggest_params(trial, space):
    """Suggest every parameter from the shared search space, so suggestions and
    the watchdog's create_trial() can never drift out of sync."""
    params = {}
    for name, dist in space.items():
        if isinstance(dist, IntDistribution):
            params[name] = trial.suggest_int(name, dist.low, dist.high, step=dist.step)
        elif isinstance(dist, CategoricalDistribution):
            params[name] = trial.suggest_categorical(name, list(dist.choices))
        else:
            raise TypeError(f"Unsupported distribution for {name}: {dist!r}")
    return params


# ==========================================
# --- MODULE 2: STATE MANAGEMENT ---
# ==========================================
def write_state(status, params, trial_number=None):
    """Durably record the current trial so a BSOD mid-benchmark can be detected on the next launch.

    Written to a temp file, fsync'd, then atomically swapped in: without the fsync the
    'pending' write may still be in the OS cache when the machine crashes, and without
    the swap a crash mid-write leaves a corrupt state.json.
    """
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"status": status, "trial": trial_number, "params": params}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_FILE)


def read_state():
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WATCHDOG] ⚠️  Could not read {STATE_FILE} ({e}); ignoring it.")
        return None
    if not isinstance(state, dict):
        print(f"[WATCHDOG] ⚠️  {STATE_FILE} has an unexpected format; ignoring it.")
        return None
    return state


def check_previous_crash(study, settings):
    """If the last run died while a trial was 'pending', the machine crashed on those settings."""
    state = read_state()
    if not state or state.get("status") != "pending":
        return

    params = state.get("params")
    print("\n[WATCHDOG] 🚨 SYSTEM CRASH DETECTED ON PREVIOUS RUN!")
    print(f"[WATCHDOG] Penalizing dangerous settings from trial {state.get('trial')}: {params}")

    try:
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=settings.space,
                value=settings.crash_penalty,
                user_attrs={"failure": "system crash (detected by watchdog)",
                            "crashed_trial": state.get("trial")},
            )
        )
        print("[WATCHDOG] Recovery complete. Resuming optimization...\n")
    except (ValueError, TypeError, KeyError) as e:
        # Usually means config.json bounds changed since that run, so the old params
        # no longer fit the search space. Don't block startup over it.
        print(f"[WATCHDOG] ⚠️  Could not record the penalty ({e}). Skipping it.\n")

    # Clear 'pending' either way so the same crash isn't penalized on every launch.
    write_state("recovered", {}, state.get("trial"))


# ==========================================
# --- MODULE 3: BENCHMARKING ---
# ==========================================
def _is_finite_number(v):
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def run_benchmark(label):
    """Run the benchmark once and turn its (avg_fps, min_fps) result into a single score.

    run_cyberpunk2077() handles crashes, hangs, a missing executable and missing/bad
    telemetry itself and reports all of them as (0.0, 0.0), so that tuple is treated
    as a failed run.
    """
    try:
        result = run_cyberpunk2077()
    except (FileNotFoundError, PermissionError) as e:
        # The harness can't even start: not the GPU settings' fault.
        raise FatalError(f"Benchmark harness error during {label} run: {e}") from e
    except Exception as e:
        raise BenchmarkFailed(f"{label} run failed: {e!r}") from e

    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise BenchmarkFailed(f"{label} run returned {result!r}, expected (avg_fps, min_fps)")
    avg_fps, min_fps = result

    if not (_is_finite_number(avg_fps) and _is_finite_number(min_fps)):
        raise BenchmarkFailed(f"{label} run returned non-numeric FPS values: {result!r}")

    if avg_fps == 0 and min_fps == 0:
        raise BenchmarkFailed(f"{label} run crashed, hung, or produced no telemetry (0, 0)")

    if avg_fps <= 0 or min_fps < 0 or min_fps > avg_fps:
        raise BenchmarkFailed(f"{label} run returned implausible FPS values: "
                              f"avg={avg_fps!r}, min={min_fps!r}")

    # Same stability-penalized metric as the harness: average FPS minus half the dip to the minimum.
    score = avg_fps - (avg_fps - min_fps) / 2.0
    return float(score)


def best_real_value(study, crash_penalty):
    """Best score from a trial that actually ran, or None if there isn't one yet."""
    try:
        best = study.best_value
    except ValueError:
        return None
    return None if best <= crash_penalty else best


# ==========================================
# --- MODULE 4: OBJECTIVE ---
# ==========================================
class Objective:
    def __init__(self, settings):
        self.s = settings
        self.bench_failures_in_a_row = 0

    def __call__(self, trial):
        params = suggest_params(trial, self.s.space)
        write_state("pending", params, trial.number)

        # Only a real crash (BSOD, power loss) leaves 'pending' behind. Every handled path
        # sets its own status; Ctrl+C or an unexpected exception is recorded as 'aborted'
        # so the watchdog doesn't penalize settings that never actually crashed.
        status = "aborted"
        try:
            # 1. Apply hardware settings
            try:
                apply_gpu_settings(trial.number, params)
            except InvalidConfig as e:
                status = "pruned"
                trial.set_user_attr("failure", f"invalid settings: {e}")
                print(f"[Trial {trial.number}] ⏭️  Pruned, settings out of driver range: {e}\n")
                raise optuna.TrialPruned(str(e))
            except UnstableConfig as e:
                status = "penalized"
                return self._penalize(trial, f"driver rejected settings: {e}")

            # 2. Benchmark (with verification for top contenders)
            voltage_penalty = abs(params["voltage"]) * self.s.voltage_penalty_weight
            try:
                raw_score = self._score(trial, voltage_penalty)
            except BenchmarkFailed as e:
                self.bench_failures_in_a_row += 1
                if self.bench_failures_in_a_row >= MAX_BENCH_FAILURES_IN_A_ROW:
                    raise FatalError(
                        f"The benchmark failed {self.bench_failures_in_a_row} trials in a row "
                        f"(last: {e}). The harness is probably broken; check it at stock settings."
                    ) from e
                status = "penalized"
                return self._penalize(trial, f"benchmark unstable: {e}")
            self.bench_failures_in_a_row = 0

            # 3. Final metric
            weighted = raw_score - voltage_penalty
            status = "completed"
            print(f"[Trial {trial.number}] Final Evaluated Score: {weighted:.4f} "
                  f"(Voltage offset: {params['voltage']}mV)\n")
            return weighted
        finally:
            write_state(status, params, trial.number)

    def _score(self, trial, voltage_penalty):
        scout = run_benchmark("scout")

        # Compare like with like: the study stores *weighted* scores.
        best = best_real_value(trial.study, self.s.crash_penalty)
        if best is not None and (scout - voltage_penalty) < best - abs(best) * VERIFY_WITHIN:
            print(f"[Trial {trial.number}] Score {scout:.2f} is not a top contender. Skipping verification.")
            return scout

        print(f"[Trial {trial.number}] 🎯 High score detected ({scout:.2f}). Running verification passes...")
        runs = [scout] + [run_benchmark(f"verification {i}") for i in range(1, VERIFY_RUNS + 1)]
        median = statistics.median(runs)
        trial.set_user_attr("runs", runs)
        print(f"[Trial {trial.number}] Verification complete. Runs: {sorted(runs)}. Settled on median: {median:.2f}")
        return median

    def _penalize(self, trial, reason):
        trial.set_user_attr("failure", reason)
        print(f"[Trial {trial.number}] 💥 {reason}. Scoring {self.s.crash_penalty}.\n")
        return self.s.crash_penalty


# ==========================================
# --- MAIN ---
# ==========================================
def print_summary(study, settings):
    print("\n=== OPTIMIZATION SUMMARY ===")
    stable = [
        t for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if t.value is not None and t.value > settings.crash_penalty
    ]
    if not stable:
        print("No stable configuration found yet.")
        return
    best = max(stable, key=lambda t: t.value)
    print(f"Best Stable Settings Found (trial {best.number}):")
    for key, value in best.params.items():
        print(f"  {key}: {value}")
    print(f"Best Metric Score: {best.value:.4f}")


def main():
    print("=== Auto Overclocker Optimization Brain ===")

    try:
        settings = load_config()
    except ConfigError as e:
        print(f"[CONFIG] ❌ {e}")
        return 2

    # Heartbeats let Optuna mark trials orphaned by a crash as FAIL on the next launch,
    # instead of leaving them stuck as RUNNING in the database forever.
    storage = optuna.storages.RDBStorage(url=DB_URL, heartbeat_interval=60, grace_period=180)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )

    check_previous_crash(study, settings)

    exit_code = 0
    try:
        print("\nStarting TPE Optimization Loop...")
        study.optimize(Objective(settings), n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\nOptimization paused by user.")
    except FatalError as e:
        print(f"\n[FATAL] ❌ {e}")
        print(f"Stopping. Progress so far is saved in {DB_URL}.")
        exit_code = 1

    print_summary(study, settings)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())