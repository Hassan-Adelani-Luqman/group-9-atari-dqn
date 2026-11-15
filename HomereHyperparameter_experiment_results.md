# Hyperparameter Experiments: Homere

**Member Name:** Homere Singizwa

Environment: ALE/Pong-v5, 300k steps, seed 2, batch size 32, epsilon 1.0 → 0.01 over 100k steps. Reward range is -21 to +21; the mean is over the last 20 episodes.

| Exp | Learning rate | Gamma | Batch size | Eps start | Eps end | Eps decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 16.25 (best 16.25); started improving at ~153k steps; mean-20 turned positive at ~280k steps, so it was still improving when training ended. Good baseline, but slow to start winning. |
| 2 | 1e-4 | 0.80 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 -16.95 (best -16.1); started improving at ~196k steps; never above 0. Gamma too low, so the agent is short-sighted and cannot link paddle hits to the point scored at the end of a rally. |
| 3 | 1e-4 | 0.90 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 -14.65 (best -13.9); started improving at ~117k steps; never above 0. Still too short-sighted, slightly better than gamma 0.8. |
| 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 -11.85 (best -11.1); started improving at ~106k steps; never above 0. Improving with gamma, but too slow at this learning rate: gamma 0.95 looks about 20 steps ahead, and at lr 1e-4 the agent had not learned enough to win in 300k steps (exp 8, same gamma with lr 2.5e-4, reached +18.95). |
| 5 | 1e-4 | 0.97 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 5.6 (best 5.6); started improving at ~117k steps; turned positive at ~270k steps, still rising at the end. Gamma 0.97 looks about 33 steps ahead, enough to start winning, but with the low learning rate it learned slowly. |
| 6 | 1e-4 | 0.995 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 11.7 (best 11.85); started improving at ~111k steps; turned positive at ~263k steps. Fairly stable at the end. Gamma 0.995 looks about 200 steps ahead, twice the baseline horizon, yet it ended below the baseline gamma 0.99 (16.25), so a longer horizon did not help at this learning rate. |
| 7 | 1e-4 | 0.999 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 7.3 (best 7.3); started improving at ~118k steps; turned positive earliest of the lr 1e-4 runs (~235k) but oscillating (last episodes 20, 15, -7). Gamma very close to 1 may make the targets noisier. |
| 8 | 2.5e-4 | 0.95 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 18.95 (best 19.05); started improving at ~70k steps; turned positive at ~170k steps and stayed near the maximum. BEST. Same gamma as exp 4, but the higher learning rate went from losing to winning. |
| 9 | 5e-5 | 0.999 | 32 | 1.0 | 0.01 | 100000 | Final mean-20 -11.0 (best -11.0); started improving at ~190k steps; never above 0. Learning rate too low to learn enough within 300k steps (same gamma as exp 7). |
| 10 | 2.5e-4 | 0.995 | 32 | 1.0 | 0.01 | 100000 | Best gamma I changed (exp 6) + Hassan's best learning rate. Final mean-20 17.2 (best 18.1 at ~254k); started improving at ~71k steps; passed 0 at ~176k and +15 at ~218k; fairly stable at the end (mean-20 between 16.9 and 17.35 over the last 5 episodes). Slightly below exp 8 (gamma 0.95, +18.95) and Hassan's exp 4 (gamma 0.99, +19.4), so the longer horizon did not help once the learning rate was higher. The big gain over exp 6 (+11.7, same gamma, lr 1e-4) came from the learning rate. |

## Best, worst and final choice

**Best:** exp 8 (γ 0.95, lr 2.5e-4): final mean-20 18.95, best 19.05. It passed 0 at ~170k steps, the earliest of my runs. Exp 10 (γ 0.995, lr 2.5e-4) was close behind at 17.2.

**Worst:** exp 2 (γ 0.80, lr 1e-4): final mean-20 -16.95. It looks only about 5 steps ahead, too short for Pong, where the reward comes at the end of a rally.

**What gamma does:** the horizon is about 1/(1-γ) steps: 0.80 ≈ 5, 0.90 ≈ 10, 0.95 ≈ 20, 0.97 ≈ 33, 0.99 ≈ 100, 0.995 ≈ 200, 0.999 ≈ 1000 (each step is 4 frames). At lr 1e-4, every γ of 0.95 or below stayed below 0, while 0.97 to 0.999 all turned positive. 0.999 oscillated.

**The learning rate mattered more than gamma in this range:**
- With γ 0.95, raising lr from 1e-4 to 2.5e-4 took exp 4 (-11.85) to exp 8 (+18.95).
- With γ 0.995, the same change took exp 6 (+11.7) to exp 10 (+17.2).
- With γ 0.999, lowering lr to 5e-5 took exp 7 (+7.3) to exp 9 (-11.0).

**Final choice:** lr 2.5e-4. At that learning rate, γ 0.95 (+18.95), 0.99 (Hassan's +19.4) and 0.995 (+17.2) all ended within about 2 points of each other, so the gap is small. I recommend γ 0.99 with lr 2.5e-4, the middle of that range and the best of the three, unless the group's comparison says otherwise. Each result is one run, and exp 8 and exp 10 used seed 2 while Hassan's used seed 1, so differences of a point or two could be noise.