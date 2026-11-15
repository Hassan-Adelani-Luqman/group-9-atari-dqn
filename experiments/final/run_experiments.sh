#!/usr/bin/env bash
# Group 9 final model: the best settings from all 30 experiments, trained 5x longer.
#
#   learning rate 2.5e-4, batch 64   Hassan: best learning rate (exp04, +19.4) and batch size (exp08, +19.5);
#                                    together in his exp10 they started improving earliest (~50k steps).
#                                    lr 2.5e-4 was also Homere's best run (exp08, +18.95).
#   gamma 0.99                       Homere: best of his runs that changed only gamma (exp01, +16.25).
#   epsilon 1.0 -> 0.01 over 250k    Kevin: his best run (exp05, +17.15) and his recommendation for a long run.
#   1,500,000 steps                  5x the 300k used in every experiment.
#
# On Kaggle (about 2h45m):  bash kaggle/launch.sh final 1 final-model
# Afterwards: bash kaggle/fetch.sh <job>, then copy experiments/final/exp01/dqn_model.pth to the repo root.
set -e
cd "$(dirname "$0")/../.."

uv run train.py --member final --exp 1 --seed 1 --lr 2.5e-4 --batch-size 64 --eps-decay 250000 --total-steps 1500000
