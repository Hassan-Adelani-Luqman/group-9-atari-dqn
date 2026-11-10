# Group 9: DQN on Atari (Formative 3) — Project Plan

> **How to use this plan:** carry out **one phase at a time**, in order (Phase 0 → Phase 7). Each phase ends with "done when" checks, and the next phase starts only after they pass. The **Task distribution** section says who does what in each phase; the Reference sections (A–D) are the shared design every phase relies on.

## Context
Group 9 has 3 members. The group must build a Deep Q-Network **from scratch** with PyTorch and Gymnasium, without SB3, RLlib, CleanRL or TorchRL, and train it on one Atari game. Each member runs **10 hyperparameter experiments, so 30 in total**. The trained agent is shown live with `play.py`, and the group presents for 10 minutes followed by 10 minutes of Q&A.

The repo is empty. The local laptop has no GPU, 4 cores, about 3 GB of free RAM and no `uv`. Because of that, **training runs on free Kaggle/Colab GPUs**. Local machines are used to write code, run smoke tests and give the live demo; inference runs fine on a CPU.

Rubric weighting that shapes this plan (30 pts in total):
- **Q&A understanding:** 10 pts
- **Collaboration evidence:** 10 pts
- **Hyperparameter documentation:** 5 pts
- **play.py and agent strength:** 5 pts

Code must be simple and easy to explain, and over-engineering is penalised.

Assumed deadline is **Sunday 15 Nov 2026, 23:59** (today is Mon 9 Nov).

**Members:**
- **Member 1: Hassan Adelani**, DQN core lead
- **Member 2: Homere Singizwa**, Env, evaluation & tooling
- **Member 3: Kevin Kaneza Mbonimpaye**, Docs, integration & final run

Folder and file names use first names so paths have no spaces, e.g. `experiments/Hassan/` and `HassanHyperparameter_experiment_results.md`.

---

## Task distribution

This section is the source for `Task allocation sheet.pdf` (filled in during Phase 0 and finalised in Phase 5). Every member gets one code area, 10 experiments, one results file, at least two README sections and about 4 minutes of the presentation.

### Roles at a glance
| | **Member 1, Hassan Adelani: DQN core lead** | **Member 2, Homere Singizwa: Env, evaluation & tooling** | **Member 3, Kevin Kaneza Mbonimpaye: Docs, integration & final run** |
|---|---|---|---|
| Setup | GitHub repo, collaborators, `uv init`, deps, `.gitignore`; Kaggle CLI runner (`kaggle/`) | Kaggle speed benchmark (first baseline run) | Books presentation slot, copies allocation sheet |
| Code | `QNetwork`, `ReplayBuffer`, `get_epsilon`, `select_action`, `train_step`, main loop | `make_env`, `play.py` (+ `--record`), CSV logging, `save_training_plots`, `plot_experiments.py` | Reviews every code PR for readability; `run_experiments.sh` files; results `.md` templates |
| Experiments | 10 × learning rate / batch size | 10 × gamma | 10 × epsilon + **final 1.5M-step run** |
| Docs | Own results `.md`; README "Training process" + "How to run" | Own results `.md`; video + GIF; README "Gameplay demo" + "Hyperparameter summary" | Own results `.md`; README intro, Mermaid diagrams, contributions; `Task allocation sheet.pdf` |
| Presentation | DQN implementation (2 min) + own experiments (2 min) | Introduction (1 min) + own experiments (2 min) | Own experiments (2 min) + **live demo** (2 min) |
| Q&A lead | Bellman target, target network, replay buffer, loss | Preprocessing, frame stacking, reward structure, env | Exploration, hyperparameters; runs the mock Q&A |

### Member 1, Hassan Adelani: DQN core lead
| Phase | Tasks | Deliverable |
|---|---|---|
| 0 | Create the GitHub repo and add both teammates. Run `git init`, `uv init` (Python 3.12) and `uv add` for the deps. Add `.gitignore` and the folder skeleton. Protect `main` so merges need 1 approving review. Push the first commit. | Repo that installs with `uv sync` |
| 1 | Write the `train.py` skeleton first (Monday) so Homere can build on it. Then add `parse_args`, `QNetwork`, `ReplayBuffer`, `get_epsilon`, `select_action`, `train_step`, the main loop, target-network sync, best-checkpoint saving and `config.json`, with comments on the key lines. | `train.py` core (PR) |
| 2 | Build the Kaggle CLI runner (`kaggle/job.py`, `launch.sh`, `fetch.sh`) and send it, with setup steps, to Homere and Kevin. Check tensor shapes and run the local 20k-step smoke test. Fix any bug the Kaggle baseline exposes. Create and push the `v1-experiments` tag. | Runner + frozen, tagged code |
| 3 | Run exp01–exp10 (learning rate and batch size). Fill in `HassanHyperparameter_experiment_results.md` and produce `comparison.png`. Review Kevin's experiments PR. | 10 models + results `.md` |
| 4 | Bring the best learning rate and batch size to the final-config decision. | Input to final config |
| 5 | Write the README "Training process" and "How to run" sections. | README sections |
| 6 | Confirm `uv run train.py` still works in a fresh clone. | Checked |
| 7 | Make the DQN-implementation slides and the slides for your own experiments. Lead Q&A on Bellman, the target network, replay and the loss. | 4 min of the talk |

### Member 2, Homere Singizwa: Env, evaluation & tooling
| Phase | Tasks | Deliverable |
|---|---|---|
| 0 | Set up and verify a Kaggle account. Run the Pong import check on your laptop. | Env works locally |
| 1 | Add `make_env`, CSV logging and `save_training_plots` to `train.py` on top of Hassan's skeleton. Write `play.py` (`--model`, `--episodes`, `--record`) and `plot_experiments.py`. | Env, logging and play PRs |
| 2 | Set up the Kaggle CLI from Hassan's instructions. Run the 300k-step baseline on Kaggle with `kaggle/launch.sh` from your own account and report steps/sec, which decides whether runs use 300k or 200k steps. Run the playback smoke test. | Step budget |
| 3 | Run exp01–exp10 (gamma). Fill in `HomereHyperparameter_experiment_results.md` and produce `comparison.png`. Review Hassan's experiments PR. | 10 models + results `.md` |
| 4 | Test the final model locally with `play.py`: greedy scores over 3 episodes. | Confirmed demo quality |
| 5 | Record `videos/gameplay.mp4` and make the GIF. Write the README "Gameplay demo" and "Hyperparameter summary" sections. | Video + README sections |
| 6 | Do the fresh-clone test on your own machine: `uv sync`, then `uv run play.py`. | Checked |
| 7 | Make the intro slides and your experiment slides. Lead Q&A on preprocessing, frame stacking, rewards and the env. Keep the backup demo video ready. | 3 min of the talk |

### Member 3, Kevin Kaneza Mbonimpaye: Docs, integration & final run
| Phase | Tasks | Deliverable |
|---|---|---|
| 0 | Book the presentation slot. Copy the allocation Google Sheet and fill it from this section. Set up a Kaggle account. | Slot booked, sheet drafted |
| 1 | Review every code PR for readability and make sure each line can be explained. Create the 3 results `.md` files from the required table template. | Reviews + templates |
| 2 | Write `run_experiments.sh` for all 3 members from the Reference D grid; each member checks their own file. | 3 run scripts |
| 3 | Run exp01–exp10 (epsilon). Fill in `KevinHyperparameter_experiment_results.md` and produce `comparison.png`. Run the Thursday sync that collects everyone's best and worst results and agrees the exp10 configs. Review Homere's experiments PR. | 10 models + results `.md` |
| 4 | Run the final 1.5M-step training. Copy the model to `/dqn_model.pth` and commit `experiments/final/`. | Final model |
| 5 | Write the README intro, the Mermaid diagrams (CNN and training loop) and the contributions table. Export `Task allocation sheet.pdf`. | README sections + PDF |
| 6 | Go through the submission checklist and submit the GitHub URL. | Submission |
| 7 | Build the slide deck shell, your experiment slides and the live demo. Lead Q&A on exploration and hyperparameters. Run the timed rehearsals and the mock Q&A. | 4 min of the talk |

### Shared tasks (everyone)
- Read and understand **all** of `train.py` and `play.py`. Anyone may be asked about any line.
- Run your own 10 experiments, write your results `.md`, and write your 2-minute "best / worst / final choice" script.
- Attend the Thursday results sync and both rehearsals, and take part in Q&A.
- Commit your own work from your own GitHub account, on your own branch.

### Review and hand-off map
| PR | Author | Reviewer |
|---|---|---|
| `train.py` core | Hassan | Kevin (Homere also checks it before building on it) |
| Env, logging, `play.py`, plots | Homere | Kevin |
| Run scripts, templates, README, docs | Kevin | Hassan |
| Hassan's experiments | Hassan | Homere |
| Homere's experiments | Homere | Kevin |
| Kevin's experiments + final run | Kevin | Hassan |

Hand-offs that block other work:
- Hassan's `train.py` skeleton (Mon) → Homere's env, logging and plots.
- Both code PRs (Tue) → Homere's Kaggle benchmark → code freeze → everyone's experiments.
- All exp01–exp09 results (Thu) → exp10 choices → final config → Kevin's final run (Fri).

### Collaboration evidence (10 pts)
- Each member works on their own branch and merges through a PR that **another member reviews** (map above).
- Each member commits their own experiment folder and results `.md` from their own GitHub account.
- The README has a contributions table that matches this section.
- `Task allocation sheet.pdf` matches the work that was actually done; update it in Phase 5 if roles shifted.

---

## Reference A: Environment choice, `ALE/Pong-v5`

| Why Pong | Detail |
|---|---|
| Learns reliably with plain DQN on a small budget | A Nature-CNN DQN reaches about +18 to +21 within roughly 0.5–1.5M agent steps, which is about 1–2 h on a T4. Most other games need 10M+ steps. |
| Simplest reward structure to explain | +1 when the agent scores and −1 when the opponent scores, so an episode is worth −21 to +21. No reward clipping is needed and the curves are easy to read. |
| No game-specific hacks | There is no FIRE-to-start and no lives handling (Breakout needs both), so there is less code to defend in Q&A. |
| Experiments that are easy to tell apart | Good configs climb from −21 to positive scores. Bad configs stay flat or collapse. |
| Strong live demo | The agent visibly beats the built-in opponent. |

MuJoCo is ruled out because its actions are continuous, and DQN needs discrete actions. Pong has 6.

**Env setup.** train.py and play.py use the same setup:
1. `gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0, render_mode=...)`
2. `AtariPreprocessing(noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False)`
3. `FrameStackObservation(stack_size=4)`

This gives a `(4, 84, 84)` uint8 state. Call `gym.register_envs(ale_py)` before creating the env. The Gymnasium wrappers are allowed because they only preprocess frames; they are not a DQN implementation.

## Reference B: Repository layout
```
group-9-atari-dqn/
├── Plan.md
├── pyproject.toml / uv.lock / .python-version (3.12) / .gitignore
├── train.py                     # full DQN implementation + training loop
├── play.py                      # loads dqn_model.pth, greedy play, real-time render
├── plot_experiments.py          # overlays one member's 10 runs (for slides/README)
├── kaggle/                      # run experiments on Kaggle GPUs from the terminal
│   ├── job.py                   # runs on Kaggle: install uv, clone repo, run selected experiments
│   ├── launch.sh                # send a job:   bash kaggle/launch.sh <Name> 1-9 v1-experiments
│   ├── fetch.sh                 # get results:  bash kaggle/fetch.sh <job>
│   └── logs/                    # full Kaggle log of every job
├── dqn_model.pth                # FINAL best model (copied from experiments/final/)
├── experiments/
│   ├── Hassan/
│   │   ├── run_experiments.sh   # the exact 10 commands (reproducibility)
│   │   ├── comparison.png
│   │   └── exp01/ … exp10/      # dqn_model.pth, config.json, training_log.csv, training_plots.png
│   ├── Homere/ …   Kevin/ …
│   └── final/                   # best config, long run → documented training performance
├── HassanHyperparameter_experiment_results.md   (exact required names, one per member)
├── HomereHyperparameter_experiment_results.md
├── KevinHyperparameter_experiment_results.md
├── videos/gameplay.mp4 (+ gameplay.gif)
├── README.md
└── Task allocation sheet.pdf
```
Install dependencies with `uv add` only, never pip: `torch gymnasium ale-py opencv-python-headless numpy matplotlib moviepy`. Python is pinned to 3.12. Torch comes from PyTorch's **CUDA 12.6** index (`[tool.uv.sources]` in `pyproject.toml`). The default PyPI build uses CUDA 13, which drops older GPUs and needs NVIDIA driver 580+. The cu126 build works on Kaggle's T4 and Colab's T4 with older drivers, and runs on CPU locally. Headless OpenCV avoids `libGL` errors on cloud machines. Each saved model is about 6.5 MB because only the online network's `state_dict` is stored. All 31 models together come to about 200 MB, which needs no Git LFS.

## Reference C: Code design (aim for train.py ≈ 200 lines, play.py ≈ 60 lines)

**`train.py`.** Small focused functions plus two tiny classes:

| Piece | What it does (step number from the assignment) |
|---|---|
| `parse_args()` | Reads `--member --exp --seed --lr --gamma --batch-size --eps-start --eps-end --eps-decay --total-steps` |
| `make_env(render_mode=None)` | Builds the env in Reference A. play.py imports it so both scripts use the same env. (step 1) |
| `class QNetwork(nn.Module)` | Nature CNN: Conv(4→32, 8, s4) → Conv(32→64, 4, s2) → Conv(64→64, 3, s1) → Flatten(3136) → FC 512 → FC n_actions, with ReLU in between. It divides the input by 255 inside `forward`. (step 2) |
| Online + target nets | The target net starts as a copy: `target_net.load_state_dict(online_net.state_dict())` (step 3) |
| `class ReplayBuffer` | Holds a `deque(maxlen=capacity)` of `(s, a, r, s', done)`. `add()` stores a transition; `sample()` uses `random.sample` and returns numpy arrays. (step 4) |
| `get_epsilon(step, start, end, decay_steps)` | Decays ε linearly from start to end over `decay_steps`, then holds it at end (step 5) |
| `select_action(...)` | Takes a random action with probability ε, otherwise the argmax Q action (step 5) |
| `train_step(...)` | Samples a batch, gets Q(s,a) from the online net, computes `y = r + γ·(1−done)·max Q_target(s')` under `no_grad`, uses Huber loss, then runs `zero_grad → backward → clip_grad_norm_(10) → optimizer.step()` (step 7) |
| Main loop | Acts, stores the transition, trains every step after warm-up, and hard-copies the target net every 1,000 steps (steps 6–8) |
| Logging | Writes one CSV row per episode: `episode, total_steps, episode_length, episode_reward, mean_reward_last20, epsilon, mean_loss` (step 9) |
| `save_training_plots()` | Saves `training_plots.png` with 3 panels: reward per episode with a rolling mean, episode length, and cumulative steps |
| Saving | Saves `dqn_model.pth` whenever the 20-episode mean hits a new best, and writes `config.json` (step 10) |

Implementation notes:
- Store `done = terminated` so that truncated episodes still bootstrap from the next state.
- Seed `random`, `numpy`, `torch` and `env.reset(seed=…)`.
- Pick the device in the order `cuda` → `mps` → `cpu`.

**`play.py`:**
- Import the shared pieces with `from train import QNetwork, make_env`.
- Build the env with `make_env(render_mode="human")` and the network with `QNetwork(n_actions)`.
- Load the weights with `load_state_dict(torch.load(path, map_location="cpu"))` and call `.eval()`.
- Choose actions greedily with argmax under `no_grad`, with no ε, and print the score of each episode.
- Flags: `--model` (default `dqn_model.pth`), `--episodes` (default 3), and `--record`, which uses `rgb_array` with `RecordVideo` to write to `videos/`.

**`plot_experiments.py <member>`.** Overlays `mean_reward_last20` against steps for exp01–exp10 and saves it as `experiments/<member>/comparison.png`.

**Fixed settings.** These are not tuned and are the same in every run:
- Replay capacity 50,000
- Learning starts after 10,000 steps
- Target update every 1,000 steps
- 1 gradient step per env step
- Adam optimizer, Huber loss
- **300,000 steps per experiment**. Phase 2 confirms this budget.

**Group baseline:** lr 1e-4 · γ 0.99 · batch 32 · ε start 1.0 · ε end 0.01 · ε decay 100,000 steps (linear).

## Reference D: Experiment grid (each member owns one theme)
Each run changes only the listed values and keeps the rest of the baseline. Every run uses the frozen code (`v1-experiments`) and the commands in `run_experiments.sh`.

| # | Hassan: Optimisation (lr, batch) | Homere: Discount (γ) | Kevin: Exploration (ε) |
|---|---|---|---|
| 1 | baseline, seed 1 | baseline, seed 2 | baseline, seed 3 |
| 2 | lr 1e-3 | γ 0.80 | ε decay 25k |
| 3 | lr 5e-4 | γ 0.90 | ε decay 50k |
| 4 | lr 2.5e-4 | γ 0.95 | ε decay 200k |
| 5 | lr 5e-5 | γ 0.97 | ε decay 250k |
| 6 | lr 1e-5 | γ 0.995 | ε end 0.10 |
| 7 | batch 16 | γ 0.999 | ε end 0.05 |
| 8 | batch 64 | γ 0.95 + lr 2.5e-4 | ε end 0.001 |
| 9 | batch 128 | γ 0.999 + lr 5e-5 | ε start 0.5 |
| 10 | best lr + best batch | best γ + Hassan's best lr | best decay + best end |

Exp01 runs the baseline with a different seed for each member. Together these three runs show how much results vary from seed alone. Exp10 combines the best findings, including teammates' results, which is also evidence of collaboration.

Use the same format for every **Observed behaviour** cell:
> *Final mean-20 reward X (best Y); started improving at ~Z k steps; stable/oscillating/collapsed; why (e.g. "lr too high → loss spikes, Q-values diverge").*

---

# Execution phases

## Phase 0: Project setup (Mon 9 Nov) · Owners: Hassan (repo), Kevin (admin)
1. Save this plan as `Plan.md` at the repo root.
2. Run `git init`, then create the GitHub repo and add the 3 members as collaborators.
3. Install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Run `uv init` with Python 3.12, then `uv add torch gymnasium ale-py opencv-python-headless numpy matplotlib moviepy`, with torch pinned to the CUDA 12.6 index (see Reference B).
4. Add a `.gitignore` covering `.venv/`, `__pycache__/` and `*.zip`. Create the empty folders from Reference B.
5. Kevin books the presentation slot (Week 6 calendar link), copies the allocation Google Sheet, and fills in the roles from the Task distribution section.
6. Each member creates a Kaggle account and verifies a phone number, which is needed for GPU and internet access. Then each member creates a Kaggle API token (saved as `~/.kaggle/access_token`) and sets `KAGGLE_USERNAME` in their shell. No notebook upload is needed: jobs are sent from the terminal (Phase 2).

**Done when:** `uv sync` works from a fresh clone, and `uv run python -c "import gymnasium as gym, ale_py; gym.register_envs(ale_py); print(gym.make('ALE/Pong-v5').action_space)"` prints `Discrete(6)`. The first commit is pushed.

## Phase 1: Core implementation (Mon 9 – Tue 10) · Owners: Hassan (train.py), Homere (env, play.py, plots)
1. **train.py** (Reference C): `parse_args`, `make_env`, `QNetwork`, `ReplayBuffer`, `get_epsilon`, `select_action`, `train_step`, main loop, CSV logging, `save_training_plots`, best-checkpoint saving and `config.json`. Add short comments on the key lines (Bellman target, done mask, target sync).
2. **play.py** (Reference C): imports `QNetwork` and `make_env` from train.py; greedy, human-rendered play; `--model`, `--episodes`, `--record`.
3. **plot_experiments.py**, which overlays one member's 10 runs.
4. All code is merged through PRs reviewed by Kevin, who checks readability and that every line can be explained.

**Done when:** the state shape is `(4, 84, 84)` uint8, `QNetwork(6)` returns `(B, 6)`, and both scripts run without errors.

## Phase 2: Verification, benchmark and code freeze (Tue 10) · Owners: Hassan (Kaggle runner), Homere (benchmark), all
1. Run a local smoke test: `uv run train.py --member test --exp 0 --total-steps 20000`. It should write `dqn_model.pth`, `config.json`, `training_log.csv` and `training_plots.png`, and the loss should stay finite.
2. Smoke-test playback with `uv run play.py --model experiments/test/exp00/dqn_model.pth --episodes 1`. A game window should appear and the episode should finish.
3. **Kaggle CLI runner** (`kaggle/`, built by Hassan). Jobs are sent from the terminal, so nobody uploads a notebook:
   - `bash kaggle/launch.sh <Name> <experiments> [git-ref]` sends a private script job to a **T4 GPU** (Kaggle has retired the P100). The job installs uv, clones the repo at the given branch or tag, runs `uv sync --frozen`, and stops if there is no GPU. It then runs the selected `--exp` lines of `experiments/<Name>/run_experiments.sh` one after another and keeps each finished experiment folder as output. `check` instead of experiment numbers only tests the setup.
   - `uvx --with six kaggle kernels status <user>/<job>` shows progress. (`--with six` works around a missing dependency in the current Kaggle CLI.)
   - `bash kaggle/fetch.sh <job>` downloads the results into `experiments/<Name>/expNN/` and saves the full log to `kaggle/logs/`.
   - The repo must be pushed (it is public) before a job can clone it.
4. Run the baseline on Kaggle for 300k steps to measure steps per second and check that learning happens. The 20-episode mean should rise above −21 by about 150–200k steps.
   - If it stays flat, check the `(1−done)` mask, the target sync, the /255 scaling, `gather`/`argmax` dims, and the ε decay.
   - If a run takes more than about 35 min, lower `--total-steps` to 200k **for everyone**.
5. Write each member's `run_experiments.sh` from the Reference D grid.
6. **Tag `v1-experiments` (code freeze).** No changes to the training logic after this point, so that all 30 runs are comparable.

**Done when:** the baseline learns on Kaggle, the step budget is fixed, the tag is pushed, and all 3 `run_experiments.sh` files are committed.

## Phase 3: Hyperparameter experiments (Tue 10 – Thu 12) · Owners: each member, for their own grid
1. Each member sends exp01–exp09 to Kaggle as one job: `bash kaggle/launch.sh <Name> 1-9 v1-experiments`. It runs in the background for up to 12 h, and 9 runs take about 4 h.
2. When the job finishes, run `bash kaggle/fetch.sh <job>` to put `experiments/<Name>/expNN/` into the repo. Then fill in each row of `<Name>Hyperparameter_experiment_results.md` using the observed-behaviour format.
3. **Thursday group sync:** share the best and worst results. Each member then sets their exp10 using teammates' findings and runs it.
4. Each member runs `uv run plot_experiments.py <Name>` to produce `comparison.png`.
5. Each member writes the **Best / worst / final choice** paragraph under their table, which also becomes their 2-minute script.
6. Commit from your own account on your own branch, and merge through a PR reviewed by another member.

**Done when:** there are 30 model folders, each with a model, log, plot and config. The 3 results `.md` files are complete, with every column filled for all 10 rows, and the 3 comparison plots are committed.

## Phase 4: Final model (Fri 13) · Owner: Kevin
1. The group picks the best overall config from the 30 runs.
2. Run it for **1.5M steps** in `experiments/final/` on Kaggle, which takes about 2 h. Use the same ε decay steps; only the total step count changes.
3. Copy the result to `/dqn_model.pth`.
4. Locally, run `uv run play.py --episodes 3` and record the greedy scores. The target is +15 or better.

**Done when:** `uv run play.py` wins games on a CPU laptop, and `experiments/final/training_plots.png` (episode reward, episode length, steps) is committed.

## Phase 5: Documentation and submission artifacts (Fri 13 – Sat 14) · Owners: as in Task distribution
1. Homere runs `uv run play.py --record` to write `videos/gameplay.mp4`, converts a short clip to `gameplay.gif`, and uploads the mp4 through GitHub's web editor so it plays inline.
2. Write **README.md**:
   1. Group members, `ALE/Pong-v5`, the game objective and the reward structure.
   2. DQN architecture: a Mermaid diagram of the CNN and a Mermaid flowchart of the training loop, plus an explanation of preprocessing, replay, the target net, the ε schedule and the fixed settings.
   3. A hyperparameter summary that links the 3 `.md` files and the comparison plots.
   4. The final run's training performance plots.
   5. The gameplay video or GIF.
   6. How to run: `uv sync`, `uv run train.py …`, `uv run play.py`.
   7. A contributions table.
3. Kevin exports the filled allocation sheet as `Task allocation sheet.pdf`.

**Done when:** every item in the submission checklist (Phase 6) is in the repo and the README renders correctly on GitHub.

## Phase 6: Final check and submission (Sun 15) · Owners: all
1. In a fresh clone on a different machine, run `uv sync` and then `uv run play.py`.
2. Submission checklist:
   - `train.py` and `play.py`
   - 30 models in member-named folders, plus `/dqn_model.pth`
   - 3 correctly named results `.md` files
   - training plots
   - README with all sections and the video
   - `Task allocation sheet.pdf`
   - only uv files, with no `requirements.txt` and no pip instructions
3. Merge everything to `main` and **submit the GitHub URL**.

**Done when:** the URL is submitted before 23:59.

## Phase 7: Presentation and Q&A preparation (Sat 14 – presentation day) · Owners: all
1. Prepare slides. Run order and timing (keep the whole presentation to **10 min**, because the sections listed in the brief add up to 11):

   | Section | Speaker | Time |
   |---|---|---|
   | Intro | Homere | 1 min |
   | DQN implementation | Hassan | 2 min |
   | Experiments | each member | 3 × ~1.75 min |
   | Live demo | Kevin | ~1.5 min |

2. Each member presents their 10 runs, the best and worst configs, which hyperparameters helped or hurt, and the justification for their final choice.
3. Live demo: `uv run play.py` on a laptop where `uv sync` has already run successfully, sharing the game window. Keep the README video as a backup.
4. Hold 2 timed rehearsals and a mock Q&A. Everyone must be able to answer:
   - Why use experience replay and a target network?
   - Write the Bellman target. Why multiply by `(1 − done)`? What is the difference between terminated and truncated?
   - Why use Huber loss rather than MSE? Why clip gradients?
   - Why stack 4 frames? Why grayscale, resize to 84 and skip frames?
   - What is the exploration–exploitation trade-off? Why decay ε, and why keep ε_end above 0?
   - Rewards in Pong are sparse and delayed. What effective horizon does γ give (about 1/(1−γ)), and why does γ = 0.8 fail?
   - What does it mean that DQN is off-policy? What does a Q-value represent? What is overestimation bias, and how would Double DQN fix it? (Not implemented.)
   - Explain each line of `train_step`, and defend your own best and worst configs.
5. Cameras stay on. Raise your hand before unmuting, and let teammates try to answer first.

**Done when:** both rehearsals finish within 10 min and every member has answered mock questions.
