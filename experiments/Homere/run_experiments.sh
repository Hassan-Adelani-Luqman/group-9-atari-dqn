#!/usr/bin/env bash
# Homere's 10 experiments: the discount factor gamma (Plan.md, Reference D).
#
# Each line sets only the values being tested. Everything else is the group baseline
# from train.py: lr 1e-4, gamma 0.99, batch 32, epsilon 1.0 -> 0.01 over 100k steps, 300k steps.
# Every run uses seed 2, so exp02-exp10 each differ from the exp01 baseline only in the values shown.
#
# On Kaggle:  bash kaggle/launch.sh Homere 1-9 v1-experiments   (runs the lines below one by one)
# Locally:    bash experiments/Homere/run_experiments.sh        (very slow without a GPU)
set -e
cd "$(dirname "$0")/../.."

uv run train.py --member Homere --exp 1 --seed 2                    # baseline (gamma 0.99)
uv run train.py --member Homere --exp 2 --seed 2 --gamma 0.80       # looks about 5 steps ahead
uv run train.py --member Homere --exp 3 --seed 2 --gamma 0.90
uv run train.py --member Homere --exp 4 --seed 2 --gamma 0.95
uv run train.py --member Homere --exp 5 --seed 2 --gamma 0.97
uv run train.py --member Homere --exp 6 --seed 2 --gamma 0.995
uv run train.py --member Homere --exp 7 --seed 2 --gamma 0.999      # looks about 1000 steps ahead
uv run train.py --member Homere --exp 8 --seed 2 --gamma 0.95 --lr 2.5e-4
uv run train.py --member Homere --exp 9 --seed 2 --gamma 0.999 --lr 5e-5

# exp10 = best gamma + Hassan's best learning rate, chosen at the Thursday sync.
# Fill in the values, remove the "# " in front, push, then: bash kaggle/launch.sh Homere 10 main
# uv run train.py --member Homere --exp 10 --seed 2 --gamma BEST_GAMMA --lr HASSANS_BEST_LR
