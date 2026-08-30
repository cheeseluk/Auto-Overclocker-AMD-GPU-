import optuna
import json
import os
from run_cyberpunk2077 import run_cyberpunk2077
from apply_gpu_settings import apply_gpu_settings
from optuna.distributions import IntDistribution, CategoricalDistribution

# ==========================================
# --- CONFIGURATION & PATHS ---
# ==========================================

with open("config.json", "r") as f:
    _config = json.load(f)

STATE_FILE = "state.json"
DB_NAME = "sqlite:///overclock_study.db"

# --- Tuning Weights ---
CRASH_PENALTY = _config["tuning"]["crash_penalty"]
VOLTAGE_PENALTY_WEIGHT = _config["tuning"]["voltage_penalty_weight"]

# --- Search Space Bounds ---
VOLTAGE_MIN = _config["params"]["voltage_offset_min"]
VOLTAGE_MAX = _config["params"]["voltage_offset_max"]
POWER_MIN = _config["params"]["power_percentage_offset_min"]
POWER_MAX = _config["params"]["power_percentage_offset_max"]
MEM_FREQ_MIN = _config["params"]["memory_freq_min"]
MEM_FREQ_MAX = _config["params"]["memory_freq_max"]
GPU_FREQ_MIN = _config["params"]["gpu_freq_min"]
GPU_FREQ_MAX = _config["params"]["gpu_freq_max"]

# ==========================================
# --- PARSED ENGINE SEARCH SPACE ---
# ==========================================
# Synced directly with executable expectations to avoid ValueError inside check_previous_crash
SEARCH_SPACE = {
    "freq_gpu": IntDistribution(GPU_FREQ_MIN, GPU_FREQ_MAX, step=10),
    "memfreq": IntDistribution(MEM_FREQ_MIN, MEM_FREQ_MAX, step=10),
    "mem_timing": CategoricalDistribution([0, 1]),
    "power_limit": IntDistribution(POWER_MIN, POWER_MAX, step=1),
    "voltage": IntDistribution(VOLTAGE_MIN, VOLTAGE_MAX, step=5)
}

# ==========================================
# --- MODULE 1: STATE MANAGEMENT ---
# ==========================================
def write_state(status, params):
    """Writes the current trial state to disk to survive hard crashes."""
    with open(STATE_FILE, "w") as f:
        json.dump({"status": status, "params": params}, f)

def check_previous_crash(study):
    """Checks if the system BSOD'd or crashed during the last run."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            
        if state.get("status") == "pending":
            print("\n[WATCHDOG] 🚨 FATAL CRASH DETECTED ON PREVIOUS RUN!")
            print(f"[WATCHDOG] Penalizing dangerous settings: {state['params']}")
            
            # Safe manual entry registration using synced definitions
            trial = optuna.trial.create_trial(
                params=state["params"],
                distributions=SEARCH_SPACE,
                value=CRASH_PENALTY
            )
            study.add_trial(trial)
            
            # Reset state so we don't infinitely penalize on the next launch
            write_state("recovered", {})
            print("[WATCHDOG] Recovery complete. Resuming optimization...\n")

def objective(trial):
    """Optuna objective function: suggest params, apply, bench, score."""
    
    current_params = {
        "freq_gpu": trial.suggest_int('freq_gpu', GPU_FREQ_MIN, GPU_FREQ_MAX, step=10),
        "memfreq": trial.suggest_int('memfreq', MEM_FREQ_MIN, MEM_FREQ_MAX, step=10),
        "mem_timing": trial.suggest_categorical('mem_timing', [0, 1]),
        "power_limit": trial.suggest_int('power_limit', POWER_MIN, POWER_MAX, step=1),
        "voltage": trial.suggest_int('voltage', VOLTAGE_MIN, VOLTAGE_MAX, step=5),
    }
    
    write_state("pending", current_params)

    # 1. Apply Hardware Settings
    success = apply_gpu_settings(trial.number, current_params)
    if not success:
        return CRASH_PENALTY

    # 2. First Benchmark Run (The "Scout" Run)
    raw_score = run_cyberpunk2077()
    
    # 3. Dynamic Noise Filtering (Conditional Reruns)
    try:
        best_known_score = trial.study.best_value
    except ValueError:
        # Happens on the very first successful trial (no 'best' yet)
        best_known_score = -float('inf')

    # If the scout run is within 2% of the best score, run verification passes
    if raw_score >= (best_known_score * 0.98):
        print(f"[Trial {trial.number}] 🎯 High score detected ({raw_score:.2f}). Running verification passes...")
        
        run_2 = run_cyberpunk2077()
        run_3 = run_cyberpunk2077()
        
        # Calculate the median to filter out extreme outliers
        scores = sorted([raw_score, run_2, run_3])
        verified_score = scores[1] 
        
        print(f"[Trial {trial.number}] Verification complete. Runs: {scores}. Settled on Median: {verified_score:.2f}")
        raw_score = verified_score
    else:
        print(f"[Trial {trial.number}] Score {raw_score:.2f} is not a top contender. Skipping verification.")

    # 4. Calculate Optimization Metric
    voltage_delta = abs(current_params["voltage"])
    weighted_score = raw_score - (voltage_delta * VOLTAGE_PENALTY_WEIGHT)

    write_state("completed", current_params)
    
    print(f"[Trial {trial.number}] Final Evaluated Score: {weighted_score:.4f} (Voltage offset: {current_params['voltage']}mV)\n")
    return weighted_score

if __name__ == "__main__":
    print("=== Auto Overclocker Optimization Brain ===")
    
    study = optuna.create_study(
        study_name="gpu_tuning_session", 
        storage=DB_NAME, 
        load_if_exists=True,
        direction="maximize"
    )

    check_previous_crash(study)

    try:
        print("\nStarting TPE Optimization Loop...")
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("\nOptimization paused by user.")

    print("\n=== OPTIMIZATION COMPLETE ===")
    print("Best Stable Settings Found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Metric Score: {study.best_value:.4f}")