"""
Rebuilds the optimization-history GIF straight from overclock_study.db.

Use this when the LIVE dashboard capture died mid-run (BSOD, crashed terminal, etc.)
but the Optuna study itself survived -- which it does, because SQLite storage is
written trial-by-trial as the study runs. This script replays that history after
the fact and produces a two-panel GIF:

  Top panel:    per-trial avg/min fps (gray), a gold star + banner marking the
                single best trial overall, a dotted baseline reference line showing
                your stock/manufacturer-default fps for comparison, a bold red
                step-line tracking the avg/min fps of whichever trial is ACTUALLY
                best-so-far by the real composite score (not raw fps), and red X
                markers for trials that crashed.

  Bottom panel: a "current best configuration" card -- the actual freq_gpu /
                memfreq / mem_timing / power_limit / voltage values of whichever
                trial holds the record as of that frame. Updates live as a new
                best trial takes over.

IMPORTANT -- what "best trial" means here: it's ranked by the composite score
(fps stability penalty + voltage penalty), not raw average fps. A trial can show
a higher gray dot than the red line and still not be "the best" -- that's
expected, not a bug, once you remember the score isn't just fps.

Requires the study to record `avg_fps` and `min_fps` as user_attrs, and the stock
run to be tagged `is_baseline`. Without that tag the dotted baseline line is
silently omitted.

Requirements: optuna, matplotlib, imageio  (pip install -r requirements.txt)
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no display needed for a batch job

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import optuna

# ==========================================
# --- CONFIG ---
# ==========================================
DB_PATH = "sqlite:///overclock_study.db"
STUDY_NAME = "gpu_tuning_session"

FRAME_DIR = "history_frames"
GIF_PATH = "optimization_history.gif"
FPS = 6
HOLD_LAST_FRAME_SECS = 3  # linger on the final frame so the loop doesn't feel abrupt

CRASH_VALUE = -10000  # sentinel objective value brain.py records for a fatal config

PARAM_ORDER = ["freq_gpu", "memfreq", "mem_timing", "power_limit", "voltage"]
PARAM_LABELS = {
    "freq_gpu": "Core clock",
    "memfreq": "Mem clock",
    "mem_timing": "Mem timing",
    "power_limit": "Power limit",
    "voltage": "Voltage offset",
}
PARAM_UNITS = {
    "freq_gpu": " MHz",
    "memfreq": " MHz",
    "mem_timing": "(1=fast)",
    "power_limit": "%",
    "voltage": " mV",
}

optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep console quiet during replay


@dataclass
class Incumbent:
    """Whichever trial currently holds the record, by composite score."""

    value: float
    number: int  # original Optuna trial number
    frame_index: int  # position on the re-indexed display timeline
    avg_fps: float
    min_fps: float
    params: dict


@dataclass(frozen=True)
class Baseline:
    """The stock/default run, used as a reference line."""

    number: int
    frame_index: int
    avg_fps: float


def load_valid_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """
    Keep explicit crashes and successful benchmarks; drop everything else.

    This discards leftover RUNNING/FAIL rows from interrupted sessions, which
    would otherwise appear as gaps in the timeline.
    """
    valid = []
    for trial in sorted(study.trials, key=lambda t: t.number):
        is_crash = trial.value == CRASH_VALUE
        is_success = (
            trial.value is not None
            and trial.value != CRASH_VALUE
            and "avg_fps" in trial.user_attrs
        )
        if is_crash or is_success:
            valid.append(trial)
    return valid


def is_success(trial: optuna.trial.FrozenTrial) -> bool:
    return trial.value != CRASH_VALUE and "avg_fps" in trial.user_attrs


def find_baseline(trials: list[optuna.trial.FrozenTrial]) -> Baseline | None:
    for frame_index, trial in enumerate(trials):
        if trial.user_attrs.get("is_baseline") and "avg_fps" in trial.user_attrs:
            return Baseline(trial.number, frame_index, trial.user_attrs["avg_fps"])
    return None


def set_header(fig, banner: str, subtitle: str, crash_note: str | None) -> None:
    """
    Replace the header text without leaking Text artists.

    fig.text() appends a NEW artist on every call rather than replacing the last
    one, so we track the ones we add and remove them ourselves. The suptitle is
    left alone -- fig.suptitle() updates its existing artist in place, and
    removing it from fig.texts would orphan matplotlib's internal reference and
    break tight_layout() on the following frame.
    """
    for text in getattr(fig, "_tracked_texts", []):
        text.remove()

    fig.suptitle(banner, fontsize=14, fontweight="bold", color="#7a5c00", y=0.985)
    tracked = [
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=10, color="#555")
    ]
    if crash_note:
        tracked.append(
            fig.text(
                0.5,
                0.91,
                crash_note,
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#c0392b",
            )
        )
    fig._tracked_texts = tracked


def draw_history_panel(
    ax,
    fig,
    *,
    trial_x: list[int],
    trial_avg: list[float],
    trial_min: list[float],
    incumbent_x: list[int],
    incumbent_avg: list[float],
    incumbent_min: list[float],
    visible_crashes: list[int],
    incumbent: Incumbent | None,
    baseline: Baseline | None,
    frame_index: int,
    total_frames: int,
) -> None:
    ax.clear()

    if trial_x:
        ax.vlines(
            trial_x, trial_min, trial_avg, color="#5b6b7a", alpha=0.35, linewidth=2, zorder=1
        )
        ax.scatter(
            trial_x,
            trial_avg,
            s=26,
            color="#5b6b7a",
            alpha=0.55,
            zorder=2,
            label="Avg fps (raw, per trial)",
        )
        ax.scatter(
            trial_x,
            trial_min,
            s=18,
            color="#5b6b7a",
            alpha=0.4,
            marker="_",
            linewidths=2,
            zorder=2,
            label="Min fps (raw, per trial)",
        )

    # baseline / stock reference
    if baseline is not None:
        ax.axhline(
            baseline.avg_fps,
            color="#444444",
            linestyle=":",
            linewidth=2,
            alpha=0.9,
            zorder=0,
            label=f"Stock/baseline avg fps (trial #{baseline.number})",
        )
        if baseline.frame_index <= frame_index:
            ax.scatter(
                [baseline.frame_index],
                [baseline.avg_fps],
                marker="D",
                s=80,
                color="#444444",
                zorder=5,
                label="Baseline trial",
            )

    if incumbent_x:
        ax.step(
            incumbent_x,
            incumbent_avg,
            where="post",
            color="#d7263d",
            linewidth=3.6,
            zorder=3,
            label="Best-so-far trial's avg fps",
        )
        ax.step(
            incumbent_x,
            incumbent_min,
            where="post",
            color="#d7263d",
            linewidth=2.2,
            linestyle="--",
            alpha=0.75,
            zorder=3,
            label="Best-so-far trial's min fps",
        )

    # gold star on the single best trial overall
    if incumbent is not None:
        ax.scatter(
            [incumbent.frame_index],
            [incumbent.avg_fps],
            marker="*",
            s=380,
            color="#f4c430",
            edgecolor="#6b5200",
            linewidth=1.2,
            zorder=6,
            label="\u2605 Best trial overall",
        )

    # crash markers sit near the bottom of whatever the data autoscaled to,
    # then the limits are nudged out so the X glyphs aren't clipped
    if visible_crashes:
        low, high = ax.get_ylim()
        span = (high - low) or 10.0
        ax.scatter(
            visible_crashes,
            [low + 0.05 * span] * len(visible_crashes),
            color="#c0392b",
            marker="x",
            s=110,
            linewidths=3,
            zorder=5,
            label=f"Crash ({CRASH_VALUE} penalty)",
        )
        ax.set_ylim(low - 0.05 * span, high + 0.05 * span)

    ax.set_xlabel("Valid trial timeline (nullities excluded)", fontsize=10)
    ax.set_ylabel("FPS", fontsize=10)
    ax.set_title(f"Trial {frame_index + 1}/{total_frames}", fontsize=10, color="#555")

    # Numeric x-ticks are hidden because this is a re-indexed sequence with the
    # invalid trials squeezed out; the title and banner carry the real trial IDs.
    ax.set_xticks([])
    ax.grid(alpha=0.25)

    while fig.legends:
        fig.legends[0].remove()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.015),
            columnspacing=1.2,
            handletextpad=0.5,
        )


def draw_config_card(ax, incumbent: Incumbent | None) -> None:
    ax.clear()
    ax.axis("off")

    if incumbent is None:
        ax.text(
            0.5,
            0.5,
            "Waiting for first benchmarked trial...",
            ha="center",
            va="center",
            fontsize=11,
            color="#888",
            transform=ax.transAxes,
        )
        return

    ax.text(
        0.5,
        0.92,
        "Current Best Configuration",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#333",
        transform=ax.transAxes,
    )
    for i, key in enumerate(PARAM_ORDER):
        x = (i + 0.5) / len(PARAM_ORDER)
        value = incumbent.params.get(key)
        value_str = f"{value:g}{PARAM_UNITS.get(key, '')}" if value is not None else "--"
        ax.text(
            x,
            0.55,
            PARAM_LABELS.get(key, key),
            ha="center",
            va="center",
            fontsize=10.5,
            color="#666",
            transform=ax.transAxes,
        )
        ax.text(
            x,
            0.20,
            value_str,
            ha="center",
            va="center",
            fontsize=19,
            fontweight="bold",
            color="#1a5fa8",
            transform=ax.transAxes,
        )


def reset_frame_dir() -> None:
    os.makedirs(FRAME_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(FRAME_DIR, "frame_*.png")):
        os.remove(path)


def main() -> None:
    study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
    valid_trials = load_valid_trials(study)
    total_frames = len(valid_trials)

    if not valid_trials:
        print(f"No completed or crashed trials in '{STUDY_NAME}'. Run brain.py first.")
        return

    baseline = find_baseline(valid_trials)
    if baseline is None:
        print("No trial tagged is_baseline -- rendering without a stock reference.")

    crash_frames = [i for i, t in enumerate(valid_trials) if t.value == CRASH_VALUE]
    crash_total = len(crash_frames)

    reset_frame_dir()
    fig, (ax_hist, ax_card) = plt.subplots(
        2, 1, figsize=(9.5, 8), gridspec_kw={"height_ratios": [5, 2]}
    )

    trial_x: list[int] = []
    trial_avg: list[float] = []
    trial_min: list[float] = []
    incumbent_x: list[int] = []
    incumbent_avg: list[float] = []
    incumbent_min: list[float] = []
    incumbent: Incumbent | None = None
    frame_paths: list[str] = []

    for frame_index, trial in enumerate(valid_trials):
        if is_success(trial):
            avg_fps = trial.user_attrs["avg_fps"]
            min_fps = trial.user_attrs["min_fps"]

            if incumbent is None or trial.value > incumbent.value:
                incumbent = Incumbent(
                    value=trial.value,
                    number=trial.number,
                    frame_index=frame_index,
                    avg_fps=avg_fps,
                    min_fps=min_fps,
                    params=trial.params,
                )

            trial_x.append(frame_index)
            trial_avg.append(avg_fps)
            trial_min.append(min_fps)

        if incumbent is not None:
            incumbent_x.append(frame_index)
            incumbent_avg.append(incumbent.avg_fps)
            incumbent_min.append(incumbent.min_fps)

        visible_crashes = [c for c in crash_frames if c <= frame_index]

        draw_history_panel(
            ax_hist,
            fig,
            trial_x=trial_x,
            trial_avg=trial_avg,
            trial_min=trial_min,
            incumbent_x=incumbent_x,
            incumbent_avg=incumbent_avg,
            incumbent_min=incumbent_min,
            visible_crashes=visible_crashes,
            incumbent=incumbent,
            baseline=baseline,
            frame_index=frame_index,
            total_frames=total_frames,
        )
        draw_config_card(ax_card, incumbent)

        if incumbent is None:
            banner = "Waiting for first benchmarked trial..."
        else:
            vs_baseline = (
                f"  (+{incumbent.avg_fps - baseline.avg_fps:.1f} fps vs stock)"
                if baseline is not None
                else ""
            )
            banner = (
                f"\u2605 Best trial so far: #{incumbent.number} \u2014 "
                f"{incumbent.avg_fps:.1f} avg / {incumbent.min_fps:.1f} min fps"
                f"{vs_baseline}"
            )
        crash_note = (
            f"{len(visible_crashes)}/{crash_total} crashes so far"
            if visible_crashes
            else None
        )
        set_header(
            fig, banner, "Auto Overclocker \u2014 Optimization History", crash_note
        )

        fig.tight_layout(rect=[0, 0.11, 1, 0.90])

        frame_path = os.path.join(FRAME_DIR, f"frame_{frame_index:04d}.png")
        fig.savefig(frame_path, dpi=110)
        frame_paths.append(frame_path)
        print(
            f"Rendered frame {frame_index + 1}/{total_frames} "
            f"(original trial #{trial.number})"
        )

    plt.close(fig)

    images = [imageio.imread(path) for path in frame_paths]
    images += [images[-1]] * (FPS * HOLD_LAST_FRAME_SECS)
    imageio.mimsave(GIF_PATH, images, fps=FPS, loop=0)
    best_number = incumbent.number if incumbent else None
    print(
        f"\nGIF saved to {GIF_PATH} ({len(frame_paths)} valid frames plotted, "
        f"{crash_total} crash(es) marked, best trial #{best_number})"
    )


if __name__ == "__main__":
    main()