# Hyperparameter Experiments: Homere

Environment: ALE/Pong-v5, 300k steps, seed 2, batch size 32, epsilon 1.0 → 0.01 over 100k steps. Reward range is -21 to +21; the mean is over the last 20 episodes.

| Exp | Learning rate | Gamma | Batch size | Eps start | Eps end | Eps decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 16.25 (best 16.25); mean-20 turned positive at ~280k steps, so it was still improving when training ended. Good baseline, but slow to start winning. |
| 2 | 1e-4 | 0.80 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 -16.95 (best -16.1); never above 0. Gamma too low, so the agent is short-sighted and cannot link paddle hits to the point scored at the end of a rally. |
| 3 | 1e-4 | 0.90 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 -14.65 (best -13.9); never above 0. Still too short-sighted, slightly better than gamma 0.8. |
| 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 -11.85 (best -11.1); never above 0. Improving with gamma, but too slow at this learning rate. |
| 5 | 1e-4 | 0.97 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 5.6 (best 5.6); turned positive at ~270k steps, still rising at the end. |
| 6 | 1e-4 | 0.995 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 11.7 (best 11.85); turned positive at ~263k steps. Good and fairly stable at the end. |
| 7 | 1e-4 | 0.999 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 7.3 (best 7.3); turned positive earliest of the lr 1e-4 runs (~235k) but oscillating (last episodes 20, 15, -7). Gamma very close to 1 may make the targets noisier. |
| 8 | 2.5e-4 | 0.95 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 18.95 (best 19.05); turned positive at ~170k steps and stayed near the maximum. BEST. Same gamma as exp 4, but the higher learning rate went from losing to winning. |
| 9 | 5e-5 | 0.999 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 -11.0 (best -11.0); never above 0. Learning rate too low to learn enough within 300k steps (same gamma as exp 7). |

**Best:** exp 8. **Worst:** exp 2.

**Final configuration and why:** TODO (after the group compares results).
