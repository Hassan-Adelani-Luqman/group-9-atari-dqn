"""
Training job that runs on a Kaggle T4 GPU. Sent there by kaggle/launch.sh, not run locally.

Steps: install uv, clone the group repo at GIT_REF, install the locked dependencies,
check that the GPU works, then run the selected experiments from
experiments/<MEMBER>/run_experiments.sh. Each finished experiment folder is copied
to /kaggle/working, which Kaggle keeps as the job's output (see kaggle/fetch.sh).
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

# Filled in by kaggle/launch.sh
MEMBER = "{{MEMBER}}"            # e.g. "Hassan"
EXPERIMENTS = "{{EXPERIMENTS}}"  # e.g. "1-9", "10", "2,5", or "check" (setup and GPU check only)
GIT_REF = "{{GIT_REF}}"          # branch or tag to train from, e.g. "v1-experiments"

REPO_URL = "https://github.com/Hassan-Adelani-Luqman/group-9-atari-dqn.git"
REPO_DIR = Path("/tmp/repo")      # outside /kaggle/working, so the 5 GB .venv is not saved as output
OUTPUT_DIR = Path("/kaggle/working")

os.environ["PATH"] = f"{Path.home()}/.local/bin:{os.environ['PATH']}"  # where the uv installer puts uv
os.environ["PYTHONUNBUFFERED"] = "1"  # show training progress in the Kaggle log as it happens


def sh(command, cwd=None, check=True):
    print(f"\n$ {command}", flush=True)
    return subprocess.run(command, shell=True, cwd=cwd, check=check).returncode


def parse_experiments(text):
    """Turn "1-9" or "2,5,10" into a set of experiment numbers. "check" gives an empty set."""
    numbers = set()
    for part in text.split(","):
        if part == "check":
            continue
        if "-" in part:
            first, last = part.split("-")
            numbers.update(range(int(first), int(last) + 1))
        else:
            numbers.add(int(part))
    return numbers


def selected_commands(numbers):
    """Return (experiment number, command) for each selected line of the member's run_experiments.sh."""
    script = REPO_DIR / "experiments" / MEMBER / "run_experiments.sh"
    commands = []
    for line in script.read_text().splitlines():
        match = re.search(r"--exp (\d+)", line)
        if line.startswith("uv run train.py") and match and int(match.group(1)) in numbers:
            commands.append((int(match.group(1)), line))
    return commands


def main():
    sh("nvidia-smi")
    sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
    sh(f"git clone --depth 1 --branch {GIT_REF} {REPO_URL} {REPO_DIR}")
    sh("uv sync --frozen", cwd=REPO_DIR)
    # Stop here rather than silently training on the CPU for hours.
    sh("uv run python -c \"import torch; assert torch.cuda.is_available(), 'No GPU'; "
       "print('GPU:', torch.cuda.get_device_name(0))\"", cwd=REPO_DIR)

    numbers = parse_experiments(EXPERIMENTS)
    if not numbers:
        print("\nSetup check passed. No experiments selected.")
        return

    commands = selected_commands(numbers)
    print(f"\nRunning {len(commands)} experiment(s) for {MEMBER}: {[n for n, _ in commands]}")
    failed = []
    for number, command in commands:
        if sh(command, cwd=REPO_DIR, check=False) != 0:
            failed.append(number)
        # Copy results straight away, so finished runs are kept even if a later one fails.
        exp_dir = Path("experiments") / MEMBER / f"exp{number:02d}"
        if (REPO_DIR / exp_dir).exists():
            shutil.copytree(REPO_DIR / exp_dir, OUTPUT_DIR / exp_dir, dirs_exist_ok=True)

    if failed:
        raise SystemExit(f"Experiments that failed: {failed}")
    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
