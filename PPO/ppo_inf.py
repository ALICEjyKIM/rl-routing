import argparse
import os
import sys
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

# inference 환경에 PyTorch가 없으면 바로 이해하기 쉬운 설치 안내를 띄움
try:
    import torch
    import torch.serialization
except ModuleNotFoundError as exc:
    raise SystemExit(
        "torch is not installed in this Python environment.\n"
        "Run this file with the same environment you used for training, or install PyTorch with:\n\n"
        f"  {sys.executable} -m pip install torch\n"
    ) from exc

from ppo_train import ActorCritic

# checkpoint를 안전하게 load할 때 numpy scalar 객체를 허용
torch.serialization.add_safe_globals([np._core.multiarray.scalar])


def require_gymnasium():
    # LunarLanderContinuous-v3는 gymnasium과 Box2D dependency가 필요함
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


def load_model(model_path, state_dim, action_dim, device):
    # train 때 사용한 ActorCritic 구조를 같은 state/action dimension으로 다시 생성
    model = ActorCritic(state_dim, action_dim).to(device)

    try:
        # PyTorch 버전에 따라 weights_only option 지원 여부가 달라서 fallback을 둠
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    # 저장된 parameter를 model에 넣고, inference mode로 전환
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def render_inference(
    model_path="PPO_basic_best.pth",
    episodes=5,
    video_folder="PPO_videos",
    human=False,
    seed=None,
):
    # 필요한 package를 확인하고 gym module을 가져옴
    gym = require_gymnasium()

    # 상대 경로로 들어온 model/video path는 PPO 폴더 기준으로 해석
    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(model_path):
        model_path = os.path.join(base_dir, model_path)
    if not os.path.isabs(video_folder):
        video_folder = os.path.join(base_dir, video_folder)

    # human=True면 화면에 바로 렌더링하고, 아니면 rgb_array로 받아 mp4로 저장
    render_mode = "human" if human else "rgb_array"

    try:
        # 학습 때와 같은 LunarLanderContinuous 환경 생성
        env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)
    except Exception as exc:
        raise SystemExit(
            "Failed to create LunarLanderContinuous-v3.\n"
            "This usually means the Box2D dependencies are missing.\n"
            "Install them with:\n\n"
            f"  {sys.executable} -m pip install \"gymnasium[box2d]\"\n"
        ) from exc

    if not human:
        try:
            # 모든 episode를 영상으로 남기기 위해 episode_trigger가 항상 True를 반환
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix="ppo_lunarlander",
                episode_trigger=lambda episode_id: True,
            )
        except Exception as exc:
            env.close()
            raise SystemExit(
                "Failed to start video recording.\n"
                "Install the video dependencies with:\n\n"
                f"  {sys.executable} -m pip install moviepy imageio imageio-ffmpeg\n"
            ) from exc

    # GPU가 가능하면 cuda에서 model을 돌리고, 아니면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_path,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        device,
    )

    for episode in range(episodes):
        # seed가 주어지면 episode마다 seed를 조금씩 바꿔 재현 가능한 다른 rollout을 만듦
        reset_seed = None if seed is None else seed + episode
        state, _ = env.reset(seed=reset_seed)
        episode_reward = 0.0

        for _ in range(1000):
            # 현재 state를 model input tensor로 변환
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                # inference에서는 sampling하지 않고 actor 평균(mu)을 action으로 사용
                # 이렇게 하면 같은 state에서 더 안정적이고 deterministic한 행동을 함
                mu, _, _ = model(state_tensor)

            # Gym environment는 numpy action을 받으므로 CPU numpy array로 변환
            action = mu.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # episode reward를 누적하고 다음 state로 이동
            episode_reward += reward
            state = next_state

            if done:
                break

        print(f"Inference Episode {episode + 1} | Reward: {episode_reward:.2f}")

    env.close()

    if not human:
        # human mode가 아닐 때만 RecordVideo가 mp4를 저장함
        print(f"Videos saved to: {video_folder}")


def parse_args():
    # command line에서 model 경로, episode 수, 영상 저장 위치 등을 바꿔 실행할 수 있게 함
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PPO_basic_best.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--video-folder", default="PPO_videos")
    parser.add_argument("--human", action="store_true", help="Show live pygame rendering instead of saving mp4.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_inference(
        model_path=args.model,
        episodes=args.episodes,
        video_folder=args.video_folder,
        human=args.human,
        seed=args.seed,
    )
