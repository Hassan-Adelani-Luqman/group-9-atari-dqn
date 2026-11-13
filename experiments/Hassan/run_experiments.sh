#!/usr/bin/env bash
# Hassan's 10 experiments: learning rate and batch size (Plan.md, Reference D).
#
# Each line sets only the values being tested. Everything else is the group baseline
# from train.py: lr 1e-4, gamma 0.99, batch 32, epsilon 1.0 -> 0.01 over 100k steps, 300k steps.
# Every run uses seed 1, so exp02-exp10 each differ from the exp01 baseline only in the values shown.
#
# On Kaggle:  bash kaggle/launch.sh Hassan 1-9 v1-experiments   (runs the lines below one by one)
# Locally:    bash experiments/Hassan/run_experiments.sh        (very slow without a GPU)
set -e
cd "$(dirname "$0")/../.."

uv run train.py --member Hassan --exp 1 --seed 1                    # baseline
uv run train.py --member Hassan --exp 2 --seed 1 --lr 1e-3          # learning rate 10x higher
uv run train.py --member Hassan --exp 3 --seed 1 --lr 5e-4
uv run train.py --member Hassan --exp 4 --seed 1 --lr 2.5e-4
uv run train.py --member Hassan --exp 5 --seed 1 --lr 5e-5
uv run train.py --member Hassan --exp 6 --seed 1 --lr 1e-5          # learning rate 10x lower
uv run train.py --member Hassan --exp 7 --seed 1 --batch-size 16
uv run train.py --member Hassan --exp 8 --seed 1 --batch-size 64
uv run train.py --member Hassan --exp 9 --seed 1 --batch-size 128

# exp10 = best learning rate + best batch size from exp01-exp09:
# lr 2.5e-4 (exp04, best mean-20 +19.7) and batch 64 (exp08, best mean-20 +19.5).
# Run with: bash kaggle/launch.sh Hassan 10 hassan-experiments
uv run train.py --member Hassan --exp 10 --seed 1 --lr 2.5e-4 --batch-size 64
