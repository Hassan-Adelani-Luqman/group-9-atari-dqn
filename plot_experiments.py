"""
Compare one member's 10 experiments in a single figure.

Reads experiments/<member>/exp01 ... exp10 and saves experiments/<member>/comparison.png:
one small panel per experiment, showing its mean reward over the last 20 games against
training steps. Every panel also shows the baseline (exp01) as a dashed line and the
member's other runs in light gray, so each run can be compared with the rest.

Example:
    uv run plot_experiments.py Hassan
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # plots are only saved to files, so no display is needed
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
from matplotlib.lines import Line2D  # noqa: E402

# The group baseline. A panel's title lists only the settings that differ from it.
BASELINE = {"lr": 1e-4, "gamma": 0.99, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 100_000}
SETTING_NAMES = {"lr": "lr", "gamma": "γ", "batch_size": "batch", "eps_start": "ε start", "eps_end": "ε end",
                 "eps_decay": "ε decay"}

RUN_COLOR = "#2a78d6"       # the panel's own experiment
BASELINE_COLOR = "#52514e"  # exp01, dashed
OTHER_COLOR = "#dcdbd6"     # the member's other experiments, for context
TEXT_COLOR = "#52514e"


def format_setting(key, value):
    """Write a setting the way the results tables do: lr 2.5e-4, ε decay 250k, γ 0.995."""
    if key == "lr":
        return f"{value:.1e}".replace(".0e", "e").replace("e-0", "e-")
    if key == "eps_decay":
        return f"{value // 1000}k"
    return f"{value:g}"


def load_run(exp_dir):
    """Return (title, steps in thousands, mean-20 rewards) for one experiment folder."""
    config = json.loads((exp_dir / "config.json").read_text())
    changes = [f"{SETTING_NAMES[key]} {format_setting(key, config[key])}"
               for key in BASELINE if config[key] != BASELINE[key]]
    title = f"{exp_dir.name}: {', '.join(changes) or 'baseline'}"
    with open(exp_dir / "training_log.csv") as file:
        rows = list(csv.DictReader(file))
    steps = [int(row["total_steps"]) / 1000 for row in rows]
    rewards = [float(row["mean_reward_last20"]) for row in rows]
    return title, steps, rewards


def main():
    parser = argparse.ArgumentParser(description="Plot one member's 10 experiments side by side.")
    parser.add_argument("member", help="folder name in experiments/, e.g. Hassan")
    member = parser.parse_args().member
    member_dir = Path("experiments") / member

    runs = [load_run(exp_dir) for exp_dir in sorted(member_dir.glob("exp*"))]
    _, baseline_steps, baseline_rewards = runs[0]  # exp01 is always the group baseline

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharex=True, sharey=True)
    for index, (ax, (title, steps, rewards)) in enumerate(zip(axes.flat, runs)):
        for _, other_steps, other_rewards in runs:
            ax.plot(other_steps, other_rewards, color=OTHER_COLOR, linewidth=1)
        if index > 0:  # exp01's own panel already shows the baseline in blue
            ax.plot(baseline_steps, baseline_rewards, color=BASELINE_COLOR, linewidth=1.2, linestyle="--")
        ax.plot(steps, rewards, color=RUN_COLOR, linewidth=2)

        ax.axhline(0, color=OTHER_COLOR, linewidth=0.8, zorder=0)  # above 0 = winning more points than losing
        ax.set_title(title, fontsize=10, loc="left")
        # Top-left is always empty: every run is near -21 early in training.
        ax.text(0.04, 0.95, f"final {rewards[-1]:+g}", transform=ax.transAxes, ha="left", va="top", fontsize=9,
                color=TEXT_COLOR)
        ax.set_ylim(-21.5, 21.5)
        ax.set_yticks([-21, -10, 0, 10, 21])
        ax.set_xticks([0, 100, 200, 300])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)

    legend = [Line2D([], [], color=RUN_COLOR, linewidth=2, label="this experiment"),
              Line2D([], [], color=BASELINE_COLOR, linewidth=1.2, linestyle="--", label="baseline (exp01)"),
              Line2D([], [], color=OTHER_COLOR, linewidth=1, label="other experiments")]
    fig.legend(handles=legend, loc="upper right", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"{member}: mean reward of the last 20 games during training", x=0.01, ha="left", fontsize=13)
    fig.supxlabel("Training steps (thousands)", fontsize=10, color=TEXT_COLOR)
    fig.supylabel("Mean reward of last 20 games (−21 to +21)", fontsize=10, color=TEXT_COLOR)
    fig.tight_layout()

    out_path = member_dir / "comparison.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
