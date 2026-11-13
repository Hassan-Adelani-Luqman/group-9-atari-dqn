# Kaggle setup and running experiments

How to set up the Kaggle command line on your own laptop and train your experiments on Kaggle's free T4 GPU, without uploading any notebook. Do the one-time setup once, then launch, fetch and commit.

- **One-time setup:** about 15 minutes
- **exp01–exp09:** about 4.5 hours on Kaggle, run one after another
- **Your laptop is free:** jobs keep running on Kaggle if you close it

There are two ways to run. Both run the same code on the same GPU:
- **Option A, command line** (below): launch, check and download from your terminal.
- **Option B, Kaggle website** ([jump there](#option-b-run-from-the-kaggle-website-no-command-line)): import a notebook and click **Save & Run All**. You don't need a token or the Kaggle CLI for this.

In every command below, replace `<Name>` with your name exactly as in the repo folders (`Hassan`, `Homere` or `Kevin`, capitalised). Replace `<name>` with the same name in lowercase. Replace anything in capitals, like `JOB_NAME`, with your own value.

Run every command from the repo folder (`group-9-atari-dqn`). **On Windows, use Git Bash for all of them**, because the scripts are bash scripts.

---

## One-time setup

### 1. Verify your phone number on Kaggle
On kaggle.com, click your profile picture (top right), then **Settings**, then **Phone verification**. Without it, Kaggle jobs get no GPU and no internet, and the job fails within the first minute.

### 2. Install git and uv
uv installs Python 3.12 and every package the project needs. Don't use pip for anything in this project.

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then open a new terminal so `uv` is on your PATH.

**Windows**: install Git for Windows from gitforwindows.org (it includes Git Bash). Then, in PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Then open **Git Bash** and stop Git from adding Windows line endings to the scripts:
```bash
git config --global core.autocrlf input
```

Check that both work:
```bash
uv --version
git --version
```

### 3. Get the latest code and the frozen tag
If you already have the repo:
```bash
cd group-9-atari-dqn
git checkout main
git pull
git fetch --tags
git tag
```
**Expected:** `git tag` lists `v1-experiments`. That tag is the frozen training code every experiment must use.

No copy yet?
```bash
git clone https://github.com/Hassan-Adelani-Luqman/group-9-atari-dqn.git
cd group-9-atari-dqn
```
Launching Kaggle jobs needs no local packages. You'll need `uv sync` later, for `play.py` and the plots; it downloads a few GB because of PyTorch.

### 4. Create your Kaggle API token
On kaggle.com: profile picture, then **Settings**, then the **API** section, then **Generate New Token**. Copy the token straight away, because Kaggle shows it only once. Then save it where the Kaggle CLI looks for it, keeping the quotes:
```bash
mkdir -p ~/.kaggle
printf '%s' 'PASTE_YOUR_TOKEN_HERE' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

> **Treat the token like a password.** Never commit it, paste it in the group chat, or put it inside the repo folder. `chmod 600` makes the file readable only by you.

If your Settings page only offers a **legacy API key**, it downloads a `kaggle.json` file instead. Move it to `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`; the CLI accepts that too.

### 5. Tell the scripts your Kaggle username
Your username is the part after `kaggle.com/` in your profile URL. It is not your email or display name.
```bash
echo 'export KAGGLE_USERNAME=YOUR_KAGGLE_USERNAME' >> ~/.bashrc
source ~/.bashrc
echo $KAGGLE_USERNAME
```
On a Mac the default shell is zsh, so use `~/.zshrc` instead of `~/.bashrc` in both lines.

### 6. Check that the Kaggle CLI can log in
```bash
uvx --with six kaggle kernels list --mine
```
**Expected:** a table of your Kaggle notebooks, or an empty list, with no error. The first run downloads the CLI, which takes a few seconds.

`uvx` runs the Kaggle CLI without installing it. `--with six` adds a package the current CLI release forgot to list. The repo's scripts already include both, so you only type this once.

### 7. Run a 3-minute setup check on a Kaggle GPU
This sends a short job that installs everything on Kaggle and confirms the GPU works, without training anything.
```bash
bash kaggle/launch.sh <Name> check main
```
The script prints the job name (like `group9-dqn-<name>-1113-0915`) plus the status and download commands, ready to copy. Run the status command until it says `COMPLETE`, then download the log:
```bash
uvx --with six kaggle kernels status YOUR_KAGGLE_USERNAME/JOB_NAME
bash kaggle/fetch.sh JOB_NAME
grep -o "GPU: Tesla T4" kaggle/logs/JOB_NAME.log
```
**Expected:** `fetch.sh` says "No experiment results in this job (setup check…)", which is correct for a check, and `grep` prints `GPU: Tesla T4`.

---

## Run your experiments

Your 10 commands are already written in `experiments/<Name>/run_experiments.sh`. The job reads them from the frozen `v1-experiments` tag, so **don't edit `train.py`**.

### 8. Launch exp01–exp09
```bash
bash kaggle/launch.sh <Name> 1-9 v1-experiments
```
Save the job name it prints. The nine runs go one after another, about 30 minutes each, so roughly 4.5 hours in total. Launch only once: a second launch starts a second, duplicate job and uses up your GPU hours.

### 9. Check progress
```bash
uvx --with six kaggle kernels status YOUR_KAGGLE_USERNAME/JOB_NAME
```
**Expected:** `RUNNING` for about 4.5 hours, then `COMPLETE`.

Kaggle only releases a job's files when the whole job has finished, so you can't download results part-way through. If a status check times out, run it again; it doesn't affect the job.

### 10. Download the results into the repo
```bash
bash kaggle/fetch.sh JOB_NAME
ls experiments/<Name>/exp0*
```
**Expected:** exp01 to exp09 each contain `config.json`, `dqn_model.pth` (about 6.5 MB), `training_log.csv` and `training_plots.png`. The full job log is saved in `kaggle/logs/`.

### 11. Commit from your own GitHub account
Each member commits their own results on their own branch, and another member reviews the pull request. This is our collaboration evidence.
```bash
git checkout -b <name>-experiments
git add experiments/<Name> kaggle/logs
git commit -m "Add <Name>'s exp01-exp09 results"
git push -u origin <name>-experiments
```
Open the pull request using the link `git push` prints, and ask your reviewer to review it:

| Your results | Reviewer |
|---|---|
| Hassan | Homere |
| Homere | Kevin |
| Kevin | Hassan |

### 12. Fill in your results table
Add one row per experiment to `<Name>Hyperparameter_experiment_results.md`, using the numbers in each `training_log.csv` and the plots. Write every "Observed behaviour" cell the same way:

> *Final mean-20 reward X (best Y); started improving at ~Z k steps; stable / oscillating / collapsed; why (e.g. "lr too high → loss spikes, Q-values diverge").*

---

## Later: exp10
After the group compares everyone's exp01–exp09, fill in the placeholders on the commented-out exp10 line at the bottom of your `run_experiments.sh`, remove the `# ` in front, commit and push. Then launch it from the branch you pushed to:
```bash
bash kaggle/launch.sh <Name> 10 main
```
The `v1-experiments` tag was made before the exp10 lines existed, so exp10 must run from a branch, not the tag. If you pushed the line to your experiments branch, use that branch name instead of `main`.

---

## Option B: run from the Kaggle website (no command line)
Use this if you'd rather not use the terminal for Kaggle. You still need **step 1** (phone verification), but not steps 4–7 (token, username, CLI). The notebook `kaggle/train.ipynb` downloads and runs the same runner the command line uses (`kaggle/job.py`).

1. **Download the notebook.** On GitHub, open `kaggle/train.ipynb` and click **Download raw file** (the download icon above the file).
2. **Import it into Kaggle.** On kaggle.com, click **Create**, then **New Notebook**. In the notebook, choose **File**, then **Import Notebook**, and upload the file.
3. **Turn on the GPU and internet.** In the right-hand panel (**Settings**, or **Session options**), set **Accelerator** to a **GPU T4** option and switch **Internet** on.
4. **Edit the settings cell** (the first code cell): set `MEMBER` to your name exactly as the folder (`Hassan`, `Homere` or `Kevin`). Keep `EXPERIMENTS = "1-9"` and `GIT_REF = "v1-experiments"`. For a 3-minute test first, use `EXPERIMENTS = "check"`.
5. **Run it in the background.** Click **Save Version**, choose **Save & Run All (Commit)**, then **Save**. You can close the tab; exp01–exp09 take about 4.5 hours. Don't use **Run All** in the editor for this: an interactive session stops when you close the tab.
6. **Download the results.** Go to kaggle.com, then **Your Work**, then **Code**, and open your notebook. When the version has finished, open its **Output** and download `<Name>_results.zip`.
7. **Put them in the repo.** Extract the zip in the repo folder, so the files land in `experiments/<Name>/exp01`, `exp02`, and so on.
   - Linux/macOS: run `unzip -o <Name>_results.zip` in the repo folder.
   - Windows: choose **Extract All** and pick the repo folder itself as the destination, not a new folder.
8. **Commit them.** Use step 11 above. To do it without git, use the GitHub website:
   1. Create a branch called `<name>-experiments` from the branch menu.
   2. Open `experiments/<Name>/`, choose **Add file**, then **Upload files**, and drag in the exp01–exp09 folders.
   3. Commit to your branch and open a pull request.

For **exp10**: push your exp10 line first, then run the notebook again with `EXPERIMENTS = "10"` and `GIT_REF = "main"` (or the branch you pushed it to).

---

## Time and GPU hours
Kaggle limits GPU time per account each week (usually about 30 hours). One experiment of 300k steps takes about 30 minutes on the T4.

| Job | Command | Time |
|---|---|---|
| Setup check | `bash kaggle/launch.sh <Name> check main` | ~3 min |
| exp01–exp09 | `bash kaggle/launch.sh <Name> 1-9 v1-experiments` | ~4.5 h |
| exp10 | `bash kaggle/launch.sh <Name> 10 main` | ~30 min |
| Final model (Kevin) | Command shared after the group picks the best settings | ~2.5 h |

---

## If something goes wrong

| You see | Fix |
|---|---|
| `401 Unauthorized`, or "could not find credentials" | The token is missing or wrong. Generate a new one (step 4), save it again and run `chmod 600`. |
| `Set KAGGLE_USERNAME to your Kaggle username` | Do step 5, then open a new terminal. |
| `uv: command not found` | Open a new terminal after installing uv. If it still fails, run `export PATH="$HOME/.local/bin:$PATH"`. |
| `No module named 'six'` | You ran `kaggle` directly. Use the repo scripts, or `uvx --with six kaggle …`. |
| `$'\r': command not found` (Windows) | The scripts have Windows line endings. Run `git config core.autocrlf input`, then `rm kaggle/*.sh` and `git checkout -- kaggle/`. |
| Status says `ERROR` | Run `bash kaggle/fetch.sh JOB_NAME` anyway, open `kaggle/logs/JOB_NAME.log`, search for "Error" or "Traceback", and send that part to Hassan. |
| `AssertionError: No GPU`, or `git clone` fails on Kaggle | Your phone number isn't verified (step 1), or your weekly GPU hours are used up. |
| `No line for experiment(s) [10]` | exp10 was launched from the tag. Push your exp10 line and launch from `main` or your branch. |
| Status stays `QUEUED` for a long time | Kaggle's GPU queue is busy. Leave it; it starts on its own. |

Training code is frozen at `v1-experiments`: please don't change `train.py`. Send questions about the scripts or results to Hassan.
