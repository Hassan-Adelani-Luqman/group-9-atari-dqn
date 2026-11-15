"""
Watch the trained DQN agent play Atari Pong.

Loads a saved Q-network and plays greedily: in every state the agent takes the
action with the highest Q-value. There are no random actions (epsilon = 0),
because exploration is only needed while training.

Examples:
    uv run play.py                     # final model (dqn_model.pth), 3 games in a window
    uv run play.py --episodes 1 --model experiments/Hassan/exp04/dqn_model.pth
    uv run play.py --record            # save each game as an mp4 in videos/ instead
"""

import argparse

import torch
from gymnasium.wrappers import RecordVideo

from train import QNetwork, make_env  # the same network and preprocessing as in training


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Pong.")
    parser.add_argument("--model", default="dqn_model.pth", help="path to the saved Q-network weights")
    parser.add_argument("--episodes", type=int, default=3, help="number of games to play")
    parser.add_argument("--record", action="store_true", help="save each game as an mp4 in videos/ instead of showing it")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.record:
        # rgb_array: the game is drawn into images instead of a window, and RecordVideo saves them as mp4 files.
        env = make_env(render_mode="rgb_array")
        # One frame is saved per agent step, i.e. every 4th game frame. Pong runs at 60 frames
        # per second, so 15 frames per second plays the video at real speed.
        env = RecordVideo(env, video_folder="videos", name_prefix="gameplay",
                          episode_trigger=lambda episode: True, fps=15)
    else:
        env = make_env(render_mode="human")  # opens a window and shows the game as it is played

    q_network = QNetwork(env.action_space.n)
    # map_location="cpu" lets a model trained on a GPU run on any laptop.
    q_network.load_state_dict(torch.load(args.model, map_location="cpu"))
    q_network.eval()  # inference only: no training from here on

    scores = []
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()  # state: the last 4 frames, shape (4, 84, 84)
        episode_reward, done = 0.0, False
        while not done:
            with torch.no_grad():  # no gradients needed to choose an action
                q_values = q_network(torch.as_tensor(state).unsqueeze(0))  # add a batch dimension: (1, 4, 84, 84)
            action = int(q_values.argmax(dim=1).item())  # greedy: the action with the highest Q-value
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward  # +1 when the agent scores, -1 when the opponent scores
            done = terminated or truncated
        scores.append(episode_reward)
        print(f"Episode {episode}: score {episode_reward:+.0f}")

    env.close()  # also finishes writing the last video when recording
    print(f"Average score over {args.episodes} episode(s): {sum(scores) / len(scores):+.1f}")


if __name__ == "__main__":
    main()
