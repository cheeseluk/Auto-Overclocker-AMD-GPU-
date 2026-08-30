"""
Renders an animated Pareto-front GIF straight from overclock_study.db.

The optimizer's real objective is a single scalar:

    score = avg_fps - (avg_fps - min_fps) / 2      (== (avg_fps + min_fps) / 2)

...but the *design goal* of the project is two-objective: maximize that score
while minimizing the absolute value of the voltage offset (smallest undervolt
that still buys you the performance). This script reconstructs that tradeoff
as a proper 2D Pareto frontier and animates it filling in trial-by-trial.

Frontier definition used here (skyline algorithm):
  - x axis = |voltage offset| (mV)   -> minimize
  - y axis = score                   -> maximize
  A trial is on the frontier at a given point in time if no earlier trial
  (equal-or-smaller |voltage|) has already matched or beaten its score. That
  makes the frontier a monotonically increasing staircase: "here's the best
  score achievable for a voltage budget of at most X."

Each frame = one more valid trial revealed. Frontier points get a gold ring,
dominated points fade to gray, crashed trials show as a red X at their
voltage (no y position, since a crash has no fps score).

Requires the study to record `avg_fps` and `min_fps` as user_attrs, and the
stock run to be tagged `is_baseline`. Without that tag the baseline reference
marker is silently omitted.

Requirements: optuna, matplotlib, imageio   (pip install -r requirements.txt)
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless -- no display needed for a batch job

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import optuna

# ==========================================
# --- CONFIG ---
# ==========================================
DB_PATH = "sqlite:///overclock_study.db"
STUDY_NAME = "gpu_tuning_session"

FRAME_DIR = "pareto_frames"
GIF_PATH = "pareto_front.gif"
FPS = 6
HOLD_LAST_FRAME_SECS = 3  # linger on the final frontier so the loop doesn't feel abrupt

CRASH_VALUE = -10000  # sentinel objective value brain.py records for a fatal config

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class TrialPoint:
    """One successfully benchmarked trial, in frontier coordinates."""

    number: int  # original Optuna trial number, for labelling
    score: float
    voltage: float  # |voltage offset| in mV


@dataclass(frozen=True)
class Baseline:
    """The stock/default run, used as a reference marker."""

    number: int
    score: float
    voltage: float


def compute_score(avg_fps: float, min_fps: float) -> float:
    """score = avg_fps - (avg_fps - min_fps) / 2  ==  (avg_fps + min_fps) / 2"""
    return avg_fps - (avg_fps - min_fps) / 2.0


def skyline_front(points: list[TrialPoint]) -> list[TrialPoint]:
    """
    Return the non-dominated subset (minimize voltage, maximize score),
    sorted by voltage ascending.

    A point is on the frontier iff its score beats every point with an
    equal-or-smaller voltage seen so far -- i.e. it's a new running max of
    score once sorted by voltage. Ties on voltage: process the highest-score
    one first so a lower-score tie doesn't wrongly qualify.
    """
    ordered = sorted(points, key=lambda p: (p.voltage, -p.score))
    front: list[TrialPoint] = []
    best_score = float("-inf")
    for point in ordered:
        if point.score > best_score:
            front.append(point)
            best_score = point.score
    return front


def load_valid_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """
    Keep explicit crashes and genuine successful runs; drop everything else.

    This naturally discards leftover RUNNING/FAIL rows from interrupted
    sessions, which would otherwise show up as gaps in the animation.
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


def to_point(trial: optuna.trial.FrozenTrial) -> TrialPoint:
    score = compute_score(trial.user_attrs["avg_fps"], trial.user_attrs["min_fps"])
    return TrialPoint(trial.number, score, abs(trial.params["voltage"]))


def find_baseline(trials: list[optuna.trial.FrozenTrial]) -> Baseline | None:
    trial = next(
        (
            t
            for t in trials
            if t.user_attrs.get("is_baseline") and "avg_fps" in t.user_attrs
        ),
        None,
    )
    if trial is None:
        return None
    point = to_point(trial)
    return Baseline(point.number, point.score, point.voltage)


def set_banner(fig, banner: str) -> None:
    """
    Replace the header text without leaking Text artists.

    fig.text() appends a NEW artist on every call rather than replacing the
    last one, so we track the ones we add and remove them ourselves. The
    suptitle is left alone -- fig.suptitle() updates its existing artist in
    place, and removing it from fig.texts would orphan matplotlib's internal
    reference and break tight_layout() on the following frame.
    """
    for text in getattr(fig, "_tracked_texts", []):
        text.remove()
    fig.suptitle(banner, fontsize=13, fontweight="bold", color="#7a5c00", y=0.97)
    fig._tracked_texts = [
        fig.text(
            0.5,
            0.91,
            "Auto Overclocker \u2014 Pareto Front (Score vs. |Voltage Offset|)",
            ha="center",
            fontsize=9.5,
            color="#555",
        )
    ]


def draw_legend(ax) -> None:
    """Deduplicate labels -- scatter calls repeat them across frames."""
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_handles, unique_labels = [], []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    ax.legend(unique_handles, unique_labels, loc="lower right", fontsize=8, frameon=False)


def draw_frame(
    fig,
    ax,
    *,
    revealed: list[TrialPoint],
    front: list[TrialPoint],
    crash_voltages: list[float],
    baseline: Baseline | None,
    frame_index: int,
    total_frames: int,
    y_bounds: tuple[float, float],
) -> None:
    ax.clear()

    front_numbers = {p.number for p in front}
    dominated = [p for p in revealed if p.number not in front_numbers]

    # dominated points -- faded gray, "explored but not on the frontier"
    if dominated:
        ax.scatter(
            [p.voltage for p in dominated],
            [p.score for p in dominated],
            s=34,
            color="#9aa5b1",
            alpha=0.45,
            zorder=2,
            label="Dominated trial",
        )

    # crash markers along the x-axis floor (no score to place them on)
    if crash_voltages:
        ax.scatter(
            crash_voltages,
            [y_bounds[0]] * len(crash_voltages),
            color="#c0392b",
            marker="x",
            s=100,
            linewidths=2.6,
            zorder=4,
            label=f"Crash ({CRASH_VALUE} penalty)",
        )

    # baseline / stock reference point
    if baseline is not None:
        ax.scatter(
            [baseline.voltage],
            [baseline.score],
            marker="D",
            s=90,
            color="#444444",
            zorder=5,
            label=f"Stock/baseline (trial #{baseline.number})",
        )

    # the frontier itself -- staircase line + gold rings on the qualifying trials
    if front:
        front_x = [p.voltage for p in front]
        front_y = [p.score for p in front]
        ax.step(
            front_x,
            front_y,
            where="post",
            color="#1a5fa8",
            linewidth=2.4,
            zorder=3,
            label="Pareto frontier (best score per voltage budget)",
        )
        ax.scatter(
            front_x,
            front_y,
            s=140,
            facecolors="#f4c430",
            edgecolors="#1a5fa8",
            linewidths=1.8,
            zorder=6,
            label="Frontier trial",
        )
        for point in front:
            ax.annotate(
                f"#{point.number}",
                (point.voltage, point.score),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
                color="#1a5fa8",
                fontweight="bold",
            )

    # star the single best trial on the frontier (max score overall so far)
    leader = max(front, key=lambda p: p.score) if front else None
    if leader is not None:
        ax.scatter(
            [leader.voltage],
            [leader.score],
            marker="*",
            s=420,
            color="#f4c430",
            edgecolor="#6b5200",
            linewidth=1.3,
            zorder=7,
            label="\u2605 Best trial overall so far",
        )

    ax.set_xlabel(
        "|Voltage offset| (mV) \u2014 lower is a smaller undervolt", fontsize=10
    )
    ax.set_ylabel("Score  (avg_fps \u2212 (avg_fps \u2212 min_fps)/2)", fontsize=10)
    ax.set_title(
        f"Pareto Front \u2014 Trial {frame_index + 1}/{total_frames}",
        fontsize=11,
        color="#333",
    )
    ax.set_ylim(*y_bounds)
    ax.grid(alpha=0.25)
    draw_legend(ax)

    if leader is None:
        banner = "Waiting for first benchmarked trial..."
    else:
        vs_baseline = (
            f"  (+{leader.score - baseline.score:.2f} score vs stock)"
            if baseline is not None
            else ""
        )
        banner = (
            f"\u2605 Frontier leader: #{leader.number} \u2014 "
            f"score {leader.score:.2f} @ {leader.voltage:.0f} mV{vs_baseline}"
        )
    set_banner(fig, banner)

    fig.tight_layout(rect=[0, 0, 1, 0.93])


def compute_y_bounds(
    points: list[TrialPoint], baseline: Baseline | None
) -> tuple[float, float]:
    """Fix the y-axis across all frames so the plot doesn't jump around."""
    scores = [p.score for p in points]
    if baseline is not None:
        scores.append(baseline.score)
    if not scores:
        return (0.0, 100.0)
    low, high = min(scores), max(scores)
    pad = (high - low) * 0.15 or 1.0
    return (low - pad, high + pad)


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

    all_points = [t for t in valid_trials if t.value != CRASH_VALUE]
    y_bounds = compute_y_bounds([to_point(t) for t in all_points], baseline)

    reset_frame_dir()
    fig, ax = plt.subplots(figsize=(9, 6.5))

    revealed: list[TrialPoint] = []
    crash_voltages: list[float] = []
    frame_paths: list[str] = []
    front: list[TrialPoint] = []

    for frame_index, trial in enumerate(valid_trials):
        if trial.value == CRASH_VALUE:
            crash_voltages.append(abs(trial.params["voltage"]))
        else:
            revealed.append(to_point(trial))

        front = skyline_front(revealed)

        draw_frame(
            fig,
            ax,
            revealed=revealed,
            front=front,
            crash_voltages=crash_voltages,
            baseline=baseline,
            frame_index=frame_index,
            total_frames=total_frames,
            y_bounds=y_bounds,
        )

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
    print(
        f"\nGIF saved to {GIF_PATH} ({len(frame_paths)} valid frames, "
        f"final frontier size: {len(front)})"
    )


if __name__ == "__main__":
    main()