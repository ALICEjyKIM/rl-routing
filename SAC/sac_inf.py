import argparse
import os
import sys
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

try:
    import torch
    import torch.serialization
except ModuleNotFoundError as exc:
    raise SystemExit(
        "torch is not installed in this Python environment.\n"
        "Run this file with the same environment you used for training, or install PyTorch with:\n\n"
        f"  {sys.executable} -m pip install torch\n"
    ) from exc

from sac_train import Actor, max_steps

torch.serialization.add_safe_globals([np._core.multiarray.scalar])


def require_gymnasium():
    try:
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "gymnasium is not installed in this Python environment.\n"
            "Install the Box2D version with:\n\n"
            f"  {sys.executable} -m pip install \"gymnasium[box2d]\"\n\n"
            "Then run this file again with the same Python interpreter."
        ) from exc

    return gym


def load_actor(model_path, state_dim, action_dim, action_low, action_high, device):
    actor = Actor(state_dim, action_dim, action_low, action_high).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


def render_inference(
    model_path="SAC_basic_best.pth",
    episodes=5,
    video_folder="SAC_videos",
    human=False,
    seed=None,
    stochastic=False,
):
    gym = require_gymnasium()

    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(model_path):
        model_path = os.path.join(base_dir, model_path)
    if not os.path.isabs(video_folder):
        video_folder = os.path.join(base_dir, video_folder)

    render_mode = "human" if human else "rgb_array"

    try:
        env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)
    except Exception as exc:
        raise SystemExit(
            "Failed to create LunarLanderContinuous-v3.\n"
            "This usually means the Box2D dependencies are missing.\n"
            "Install them with:\n\n"
            f"  {sys.executable} -m pip install \"gymnasium[box2d]\""
        ) from exc

    if not human:
        try:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix="sac_lunarlander",
                episode_trigger=lambda episode_id: True,
            )
        except Exception as exc:
            env.close()
            raise SystemExit(
                "Failed to start video recording.\n"
                "Install the video dependencies with:\n\n"
                f"  {sys.executable} -m pip install moviepy imageio imageio-ffmpeg\n"
            ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = load_actor(
        model_path,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        device,
    )

    for episode in range(episodes):
        reset_seed = None if seed is None else seed + episode
        state, _ = env.reset(seed=reset_seed)
        episode_reward = 0.0

        for _ in range(max_steps):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                sampled_action, _, deterministic_action = actor.sample(state_tensor)
                action = sampled_action if stochastic else deterministic_action

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated

            episode_reward += reward
            state = next_state

            if done:
                break

        print(f"Inference Episode {episode + 1} | Reward: {episode_reward:.2f}")

    env.close()

    if not human:
        print(f"Videos saved to: {video_folder}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="SAC_basic_best.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--video-folder", default="SAC_videos")
    parser.add_argument("--human", action="store_true", help="Show live pygame rendering instead of saving mp4.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the SAC policy instead of using the deterministic mean action.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_inference(
        model_path=args.model,
        episodes=args.episodes,
        video_folder=args.video_folder,
        human=args.human,
        seed=args.seed,
        stochastic=args.stochastic,
    )
