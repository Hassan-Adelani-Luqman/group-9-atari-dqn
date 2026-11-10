"""
Train a Deep Q-Network (DQN) agent to play Atari Pong.

The DQN parts are implemented from scratch with PyTorch: the Q-network,
the replay buffer, epsilon-greedy exploration, the Bellman target and the
training loop. Gymnasium only provides the game.

Example (group baseline):
    uv run train.py --member Hassan --exp 1 --seed 1

Results are written to experiments/<member>/exp<NN>/:
    dqn_model.pth, config.json, training_log.csv, training_plots.png
"""

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

gym.register_envs(ale_py)

# Fixed settings: the same in every experiment. Only the arguments in parse_args() are tuned.
ENV_ID = "ALE/Pong-v5"
REPLAY_CAPACITY = 50_000     # number of transitions kept in the replay buffer
LEARNING_STARTS = 10_000     # steps collected before the first network update
TARGET_UPDATE_EVERY = 1_000  # steps between copying the online network into the target network
MAX_GRAD_NORM = 10.0         # gradient clipping keeps single updates from being too large
REWARD_WINDOW = 20           # number of recent episodes averaged for the "mean reward" metric


def parse_args():
    """Read the run's hyperparameters from the command line (defaults = group baseline)."""
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Pong.")
    parser.add_argument("--member", required=True, help="member's first name, used for the output folder")
    parser.add_argument("--exp", type=int, required=True, help="experiment number, e.g. 1 to 10")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--eps-decay", type=int, default=100_000, help="steps to decay epsilon linearly")
    parser.add_argument("--total-steps", type=int, default=300_000)
    return parser.parse_args()


def make_env(render_mode=None):
    """Create Pong with the standard Atari preprocessing (Plan.md, Reference A).

    TODO (Homere, Phase 1):
        gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, render_mode=render_mode)
        -> AtariPreprocessing(noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False)
        -> FrameStackObservation(stack_size=4)
    Observations must have shape (4, 84, 84) and dtype uint8.
    """
    raise NotImplementedError("make_env is Homere's Phase 1 task")


def save_training_log(episode_log, out_dir):
    """Write episode_log (a list of dicts, one per episode) to out_dir/training_log.csv.

    TODO (Homere, Phase 1): one CSV row per episode, columns in the order of the dict keys.
    """


def save_training_plots(episode_log, out_dir):
    """Save out_dir/training_plots.png with 3 panels: episode reward (+ rolling mean),
    episode length and cumulative steps, each against the episode number.

    TODO (Homere, Phase 1)
    """


class QNetwork(nn.Module):
    """CNN that maps a stack of 4 frames (4 x 84 x 84) to one Q-value per action."""

    def __init__(self, n_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # -> 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> 64 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),                                # -> 3136
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),                   # one Q-value per action
        )

    def forward(self, x):
        return self.layers(x.float() / 255.0)  # scale pixels from [0, 255] to [0, 1]


class ReplayBuffer:
    """Fixed-size memory of past transitions. When full, the oldest ones are dropped."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Random sampling breaks the correlation between consecutive game frames.
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.memory)


def get_epsilon(step, eps_start, eps_end, eps_decay):
    """Decay epsilon linearly from eps_start to eps_end over eps_decay steps, then keep it at eps_end."""
    fraction = min(step / eps_decay, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def select_action(q_network, state, epsilon, n_actions, device):
    """Epsilon-greedy: a random action with probability epsilon, otherwise the highest Q-value."""
    if random.random() < epsilon:
        return random.randrange(n_actions)  # explore
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)  # add a batch dimension
        q_values = q_network(state_tensor)
    return int(q_values.argmax(dim=1).item())  # exploit


def train_step(online_net, target_net, optimizer, buffer, batch_size, gamma, device):
    """Update the online network once on a random batch from the replay buffer. Returns the loss."""
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states = torch.as_tensor(states, device=device)
    actions = torch.as_tensor(actions, device=device)
    rewards = torch.as_tensor(rewards, device=device)
    next_states = torch.as_tensor(next_states, device=device)
    dones = torch.as_tensor(dones, device=device)

    # 1. Q(s, a) from the online network, for the actions that were actually taken.
    q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # 2. Bellman target from the target network: y = r + gamma * max_a' Q_target(s', a').
    #    (1 - done) removes the future term when the episode really ended.
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1).values
        targets = rewards + gamma * (1 - dones) * next_q_values

    # 3. Huber loss: like MSE for small errors, but less sensitive to large ones.
    loss = nn.functional.smooth_l1_loss(q_values, targets)

    # 4. Backpropagation, then 5. one optimizer step on the online network.
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    return loss.item()


def log_episode(episode_log, step, episode_reward, episode_length, epsilon, episode_losses):
    """Append this episode's statistics to episode_log and return the recent mean reward."""
    recent_rewards = [row["episode_reward"] for row in episode_log[-(REWARD_WINDOW - 1):]]
    mean_reward = float(np.mean(recent_rewards + [episode_reward]))
    episode_log.append({
        "episode": len(episode_log) + 1,
        "total_steps": step,
        "episode_length": episode_length,
        "episode_reward": episode_reward,
        "mean_reward_last20": round(mean_reward, 2),
        "epsilon": round(epsilon, 4),
        "mean_loss": round(float(np.mean(episode_losses)), 5) if episode_losses else None,
    })
    return mean_reward


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_config(args, out_dir):
    """Record every setting of this run next to its results."""
    config = vars(args) | {
        "env_id": ENV_ID,
        "replay_capacity": REPLAY_CAPACITY,
        "learning_starts": LEARNING_STARTS,
        "target_update_every": TARGET_UPDATE_EVERY,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path("experiments") / args.member / f"exp{args.exp:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "dqn_model.pth"
    save_config(args, out_dir)
    print(f"Training on {device}, saving to {out_dir}")

    env = make_env()
    n_actions = env.action_space.n

    # Online network (trained every step) and target network (a periodically updated copy
    # used only to compute the Bellman targets, which keeps those targets stable).
    online_net = QNetwork(n_actions).to(device)
    target_net = QNetwork(n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(REPLAY_CAPACITY)

    episode_log = []
    best_mean_reward = -float("inf")
    episode_reward, episode_length, episode_losses = 0.0, 0, []
    state, _ = env.reset(seed=args.seed)
    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        # Act in the environment and store the experience.
        epsilon = get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
        action = select_action(online_net, state, epsilon, n_actions, device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Only `terminated` counts as done: a time-limit cut (truncated) is not a real end,
        # so the Bellman target should still use the value of the next state.
        buffer.add(state, action, reward, next_state, terminated)
        state = next_state
        episode_reward += reward
        episode_length += 1

        # Learn from a random batch of past experience once the buffer has enough of it.
        if step > LEARNING_STARTS:
            loss = train_step(online_net, target_net, optimizer, buffer, args.batch_size, args.gamma, device)
            episode_losses.append(loss)

        # Periodically copy the online weights into the target network.
        if step % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(online_net.state_dict())

        if terminated or truncated:
            mean_reward = log_episode(episode_log, step, episode_reward, episode_length, epsilon, episode_losses)
            save_training_log(episode_log, out_dir)

            # Keep the model with the best recent average, which guards against a late collapse.
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(online_net.state_dict(), model_path)

            steps_per_sec = step / (time.time() - start_time)
            print(f"Episode {len(episode_log):4d} | step {step:7d} | reward {episode_reward:+5.0f} | "
                  f"length {episode_length:5d} | mean{REWARD_WINDOW} {mean_reward:+6.2f} | "
                  f"eps {epsilon:.3f} | {steps_per_sec:.0f} steps/s")

            episode_reward, episode_length, episode_losses = 0.0, 0, []
            state, _ = env.reset()

    env.close()
    save_training_plots(episode_log, out_dir)
    print(f"Done. Best mean{REWARD_WINDOW} reward: {best_mean_reward:+.2f}. Model saved to {model_path}")


if __name__ == "__main__":
    main()
