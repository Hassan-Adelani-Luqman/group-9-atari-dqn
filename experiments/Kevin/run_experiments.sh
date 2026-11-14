#!/usr/bin/env bash
# Kevin's 10 experiments: epsilon-greedy exploration (Plan.md, Reference D).
#
# Each line sets only the values being tested. Everything else is the group baseline
# from train.py: lr 1e-4, gamma 0.99, batch 32, epsilon 1.0 -> 0.01 over 100k steps, 300k steps.
# Every run uses seed 3, so exp02-exp10 each differ from the exp01 baseline only in the values shown.
#
# On Kaggle:  bash kaggle/launch.sh Kevin 1-9 v1-experiments   (runs the lines below one by one)
# Locally:    bash experiments/Kevin/run_experiments.sh        (very slow without a GPU)
set -e
cd "$(dirname "$0")/../.."

uv run train.py --member Kevin --exp 1 --seed 3                     # baseline
uv run train.py --member Kevin --exp 2 --seed 3 --eps-decay 25000   # stop exploring early
uv run train.py --member Kevin --exp 3 --seed 3 --eps-decay 50000
uv run train.py --member Kevin --exp 4 --seed 3 --eps-decay 200000
uv run train.py --member Kevin --exp 5 --seed 3 --eps-decay 250000  # explore for most of the run
uv run train.py --member Kevin --exp 6 --seed 3 --eps-end 0.1       # keep 10% random actions
uv run train.py --member Kevin --exp 7 --seed 3 --eps-end 0.05
uv run train.py --member Kevin --exp 8 --seed 3 --eps-end 0.001     # almost no random actions
uv run train.py --member Kevin --exp 9 --seed 3 --eps-start 0.5     # start half random

# exp10 = best epsilon decay + best final epsilon, chosen at the Thursday sync.
# Fill in the values, remove the "# " in front, push, then: bash kaggle/launch.sh Kevin 10 main
# uv run train.py --member Kevin --exp 10 --seed 3 --eps-decay BEST_DECAY --eps-end BEST_END
