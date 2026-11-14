# Hyperparameter experiment results

**Member Name:** Hassan Adelani

**Focus:** learning rate and batch size. Each experiment changes only these from the group baseline; all other settings stay the same.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 100,000 steps (linear) | **Baseline.** Final mean-20 reward +18.5 (best +19.3 at 266k). Started improving at ~81k steps, passed 0 at 165k and +15 at 191k. Stable after that: the mean-20 never fell below +15.8, and it won every game after 250k. The group baseline learns reliably once the network stays healthy. |
| 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.01 | 100,000 steps (linear) | Final mean-20 reward −20.9 (best −20.3, from random play at 18k). Never improved; **collapsed**. Within about 9k updates every unit in the third conv layer and in the 512-unit layer stopped firing ("dying ReLU"). The network then output the same Q-values for every frame and the agent kept repeating one action. Steps 10× larger than the baseline pushed the units below zero for good. The loss stayed flat at about 0.012 because nothing was being learned. |
| 3 | 5e-4 | 0.99 | 32 | 1.0 | 0.01 | 100,000 steps (linear) | Final mean-20 reward −20.9 (best −19.9 at 19k). Never improved; **collapsed** the same way as exp 2. Early on, 95% of the third conv layer and 89% of the 512-unit layer were dead, and the Q-values were identical for every state. Even 5× the baseline step size is too large for this network. |
| 4 | 2.5e-4 | 0.99 | 32 | 1.0 | 0.01 | 100,000 steps (linear) | Final mean-20 reward +19.4 (**best +19.7** at 248k, the highest of my runs). Started improving at ~62k steps, earlier than any other single change. Passed 0 at 182k and +15 at 213k. Stable: never below +15.4 after that, and it won every game after 250k. 2.5× the baseline step size speeds up learning without killing units (only 3% dead). |
| 5 | 5e-5 | 0.99 | 32 | 1.0 | 0.01 | 100,000 steps (linear) | Final mean-20 reward −16.9 (best −16.8 at 289k). Started improving late (~158k steps) and was still climbing slowly at 300k. The network was healthy (no dead units); learning was simply too slow. At half the baseline step size it needs well over 300k steps. |
| 6 | 1e-5 | 0.99 | 32 | 1.0 | 0.01 | 100,000 steps (linear) | Final mean-20 reward −20.0 (best −20.0 at 294k). Essentially never improved. The loss did fall (from 0.012 at 100k to 0.003), so it was learning state values. But with steps 10× smaller than the baseline, the policy did not change enough to win points within 300k steps. |
| 7 | 1e-4 | 0.99 | 16 | 1.0 | 0.01 | 100,000 steps (linear) | Final mean-20 reward −9.0 (best −9.0 at 298k). Started improving at ~87k but climbed much more slowly than the baseline. It won only 11% of games after 250k and was still rising at the end. Each update averages only 16 transitions, so the gradient is noisier and each step less reliable. This is the likely cause, though single runs vary (see the note below). |
| 8 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 100,000 steps (linear) | Final mean-20 reward **+19.5** (best +19.5 at 299k). Started improving at ~71k, passed 0 at 183k and +15 at 207k. Stable: never below +15.4 after that, and it won every game after 250k. Averaging 64 transitions gives smoother gradients, and it matched the best learning rate. Runtime was 32 min, against 29 for batch 32. |
| 9 | 1e-4 | 0.99 | 128 | 1.0 | 0.01 | 100,000 steps (linear) | Final mean-20 reward +4.0 (best +4.1 at 292k). Started improving at ~86k but passed 0 only at 244k. It was still rising at 300k and won 60% of games after 250k. Adam takes steps of about the same size whatever the batch, so a bigger batch should not learn slower per update; most of the gap to batch 64 is probably run-to-run variation. The clear cost is time: 47 min, 1.6× the baseline. |
| 10 | 2.5e-4 | 0.99 | 64 | 1.0 | 0.01 | 100,000 steps (linear) | **Best learning rate (exp 4) + best batch size (exp 8).** Final mean-20 reward +19.3 (best +19.4 at 282k). **Started improving earliest of all runs (~50k)**, passed 0 at 171k and +15 at 216k. Stable: never below +15.8 after that, and it won every game after 250k. The fastest start, but the same final score as exp 4 and exp 8, because all three are close to Pong's +21 maximum by 300k. |

## Best, worst and final choice

- **Best:** exp 4 (learning rate 2.5e-4) reached the highest mean-20 reward, +19.7. Exp 8 (batch 64) and exp 10 (both together) were within half a point. Exp 10 started improving soonest, at about 50k steps instead of 81k for the baseline.
- **Worst:** exp 2 and exp 3 (learning rates 1e-3 and 5e-4). They never learned because the network died early: whole layers of ReLU units stopped firing, so every frame got the same Q-values. Exp 6 (1e-5) also failed, but for the opposite reason: learning was far too slow.
- **What learning rate does:** the useful range is narrow. Between 1e-4 and 2.5e-4 the agent learns well. Above that, updates are big enough to kill units permanently. Below it, learning is too slow to win games within 300k steps.
- **What batch size does:** 64 matched the best result, and 16 learned clearly more slowly. 128 learned more slowly in this run, but ran 1.6× longer, which makes it the worst trade-off.
- **Final choice:** learning rate **2.5e-4** with batch size **64** (exp 10). It learns fastest and reaches about +19.3 to +19.7, as good as any setting I tested. I recommend learning rate 2.5e-4 for the group's final model.

## How to read the table

- **Fixed settings in every run:**
  - environment `ALE/Pong-v5`, 300,000 steps, seed 1
  - replay buffer of 50,000 transitions, learning starts after 10,000 steps
  - target network copied every 1,000 steps
  - Adam optimiser (ε = 1.5e-4), Huber loss, gradient clipping at 10
- **Mean-20 reward:** the average reward of the last 20 games. A game's reward runs from −21 (lost every point) to +21 (won every point). "Started improving" is the first step where the mean-20 rose above −19.
- **Results for each run:** `experiments/Hassan/expNN/`. Each folder holds `training_plots.png` (reward, episode length and steps per episode), `training_log.csv` (one row per game), `config.json` and the saved model `dqn_model.pth`.
- **Run-to-run variation:** GPU training is not exactly repeatable. The same baseline settings and seed reached +1.4 in an earlier test job but +18.5 here. Differences of about a point between the top runs are not meaningful; the large gaps (learning or not, collapse or not) are.
