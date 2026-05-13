import csv
import os
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import torch.serialization

torch.serialization.add_safe_globals([np._core.multiarray.scalar])


# 하이퍼파라미터
# gamma: 현재 reward뿐 아니라 미래 reward를 얼마나 중요하게 볼지 정하는 할인율
gamma = 0.99
# lr: Adam optimizer가 model parameter를 한 번에 얼마나 크게 업데이트할지 정하는 learning rate
lr = 0.0003
# clip_epsilon: PPO의 핵심 값. 새 policy가 old policy에서 너무 멀리 움직이지 않도록 ratio를 제한
clip_epsilon = 0.2
# ppo_epochs: 한 episode에서 모은 rollout data를 몇 번 반복해서 학습할지 정함
ppo_epochs = 10
# entropy_coef: action 분포의 entropy 보너스 비중. 너무 빨리 한 action으로 굳는 것을 방지
entropy_coef = 0.01
# value_coef: critic loss가 전체 loss에서 차지하는 비중
value_coef = 0.5
# max_steps: 한 episode 안에서 환경과 상호작용할 최대 step 수
max_steps = 1000
# max_episodes: 전체 학습 episode 수의 최대값
max_episodes = 10000
# patience: best reward가 개선되지 않아도 기다릴 episode 수
patience = 50
# target_reward: 이 평균 reward 이상이면 LunarLander를 해결했다고 보고 early stopping 가능
target_reward = 200

# 학습 결과 파일들은 PPO 폴더 안에 저장되도록 경로를 고정
base_dir = os.path.dirname(__file__)
log_path = os.path.join(base_dir, "PPO_training_log.csv")
model_path = os.path.join(base_dir, "PPO_basic_best.pth")
plot_path = os.path.join(base_dir, "PPO_basic_metrics.png")


# ActorCritic은 actor와 critic을 하나의 neural network 안에서 같이 계산하는 구조
# actor는 action 분포의 평균(mu)과 표준편차(std)를 만들고, critic은 현재 state의 가치 V(s)를 예측
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # state vector를 128차원 feature로 변환하는 공통 feature extractor
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )

        # actor head: feature를 action_dim 크기의 평균값으로 변환
        self.mu = nn.Linear(128, action_dim)
        # action 분포의 표준편차를 직접 학습하기 위해 log_std를 parameter로 둠
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        # critic head: feature에서 state value V(s)를 하나의 scalar로 예측
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        # 입력 state를 공통 feature로 변환
        x = self.fc(x)
        # tanh를 거쳐 action 평균을 -1~1 범위로 제한
        mu = torch.tanh(self.mu(x))
        # exp(log_std)를 사용하면 std가 항상 양수가 됨
        std = torch.exp(self.log_std)
        # critic이 현재 state의 value를 출력
        value = self.value(x)
        return mu, std, value


def save_model(model, optimizer, episode, best_reward, path):
    # 학습을 다시 이어서 하거나 inference에 사용할 수 있도록 model/optimizer 상태를 checkpoint로 저장
    checkpoint = {
        "episode": episode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_reward": best_reward,
    }
    torch.save(checkpoint, path)
    print(f"--- Checkpoint saved at episode {episode} (Best: {best_reward:.2f}) ---")


def make_log_file():
    # csv log file이 없거나 비어 있으면 header를 먼저 만들어 둠
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "episode",
                    "avg_reward",
                    "best_reward",
                    "actor_loss",
                    "value_loss",
                    "entropy",
                ]
            )


def compute_returns(rewards, masks, last_value, device):
    # 각 step의 discounted return을 뒤에서부터 계산
    # 마지막 state의 value를 bootstrap 값으로 사용해서 episode가 끝나지 않은 경우도 처리
    returns = []
    R = last_value.detach()

    for reward, mask in zip(reversed(rewards), reversed(masks)):
        # done이면 mask가 0이므로 다음 value가 끊기고, 아니면 gamma * R이 이어짐
        R = reward + gamma * R * mask
        returns.insert(0, R)

    return torch.stack(returns).view(-1).to(device)


def train():
    # LunarLanderContinuous는 action이 연속값인 환경이므로 Normal distribution policy를 사용
    env = gym.make("LunarLanderContinuous-v3")
    # GPU가 가능하면 cuda를 사용하고, 아니면 CPU로 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # observation/action dimension을 환경에서 직접 읽어 model input/output 크기를 맞춤
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # sample한 action이 환경의 action 범위를 벗어나지 않도록 clamp에 사용할 bound
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)

    # ActorCritic model과 optimizer 준비
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    make_log_file()

    all_rewards, avg_rewards = [], []
    actor_losses, value_losses = [], []
    avg_actor_losses, avg_value_losses = [], []
    entropies, avg_entropies = [], []

    best_reward = -float("inf")
    no_improve_count = 0

    for episode in range(max_episodes):
        # episode 시작 시 환경 초기화
        state, _ = env.reset()

        # PPO는 한 번 rollout을 모은 뒤, 그 data를 여러 epoch 동안 재사용해서 학습
        # old_log_probs는 data를 모을 당시 old policy의 log probability로, ratio 계산에 필요
        states, actions, old_log_probs = [], [], []
        # rewards/masks/values는 return과 advantage 계산에 사용
        rewards, masks, values = [], [], []
        episode_reward = 0.0

        for _ in range(max_steps):
            # 현재 state를 tensor로 변환하고 batch dimension을 추가
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                # rollout 수집 단계에서는 gradient가 필요 없음
                mu, std, value = model(state_tensor)
                # actor가 만든 mu/std로 연속 action을 sampling
                dist = Normal(mu, std)
                action = dist.sample()
                # 나중에 PPO ratio를 계산하기 위해 old policy의 log probability 저장
                log_prob = dist.log_prob(action).sum(dim=-1)

            # 환경 action 범위에 맞춰 action을 잘라낸 뒤 numpy로 변환해 env.step에 전달
            env_action = torch.clamp(action.squeeze(0), action_low, action_high)
            next_state, reward, terminated, truncated, _ = env.step(env_action.cpu().numpy())
            done = terminated or truncated

            # rollout data 저장
            states.append(state_tensor.squeeze(0))
            actions.append(action.squeeze(0))
            old_log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))
            # done이면 0, 아직 이어지면 1. return 계산에서 episode 종료 지점을 끊는 역할
            masks.append(torch.tensor([1 - done], dtype=torch.float32, device=device))

            episode_reward += reward
            state = next_state

            if done:
                break

        with torch.no_grad():
            # 마지막 state의 critic value를 bootstrap 값으로 사용
            next_state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = model(next_state_tensor)[2]

        # list로 모아 둔 rollout data를 batch tensor로 묶음
        states = torch.stack(states)
        actions = torch.stack(actions)
        # old_log_probs와 old_values는 target처럼 쓰이므로 gradient가 흐르지 않게 detach
        old_log_probs = torch.stack(old_log_probs).detach()
        old_values = torch.stack(values).view(-1).detach()
        returns = compute_returns(rewards, masks, last_value, device)

        # advantage = 실제로 얻을 것으로 계산된 return - critic이 예측했던 value
        advantages = returns - old_values
        # advantage를 정규화하면 학습이 더 안정적이고 scale에 덜 민감해짐
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        last_actor_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0

        for _ in range(ppo_epochs):
            # 같은 rollout data를 사용하지만, 현재 model 기준의 log probability/value를 다시 계산
            mu, std, new_values = model(states)
            dist = Normal(mu, std)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            # entropy가 높으면 다양한 action을 시도한다는 뜻. exploration 유지용 보너스로 사용
            entropy = dist.entropy().sum(dim=-1).mean()

            # PPO ratio = 현재 policy가 old policy에 비해 같은 action을 얼마나 더/덜 선택하는지
            ratios = torch.exp(new_log_probs - old_log_probs)
            # unclipped objective와 clipped objective 중 더 보수적인 값을 사용
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

            # actor는 clipped surrogate를 최대화해야 하므로 loss에서는 음수로 바꿈
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            # critic은 predicted value가 returns와 가까워지도록 MSE로 학습
            value_loss = F.mse_loss(new_values.view(-1), returns)
            # 전체 loss = actor loss + critic loss - entropy bonus
            loss = actor_loss + value_coef * value_loss - entropy_coef * entropy

            # 이전 gradient 초기화
            optimizer.zero_grad()
            # actor, critic 쪽 파라미터가 각 loss에 따라 얼마나 바뀌어야 하는지 계산
            # PyTorch에서는 total loss 하나로 backward를 한 번 호출해도 actor/critic 양쪽에 각각 gradient가 계산됨
            loss.backward()
            # 계산된 gradient를 이용해 실제 actor/critic 파라미터 업데이트
            optimizer.step()

            # 기록용
            # 마지막 PPO epoch의 loss 값을 logging용으로 저장
            last_actor_loss = actor_loss.item()
            last_value_loss = value_loss.item()
            last_entropy = entropy.item()

        # reward/loss/entropy를 저장하고 최근 10개 episode 이동평균을 계산
        all_rewards.append(episode_reward)
        avg_reward = np.mean(all_rewards[-10:])
        avg_rewards.append(avg_reward)

        actor_losses.append(last_actor_loss)
        avg_actor_loss = np.mean(actor_losses[-10:])
        avg_actor_losses.append(avg_actor_loss)

        value_losses.append(last_value_loss)
        avg_value_loss = np.mean(value_losses[-10:])
        avg_value_losses.append(avg_value_loss)

        entropies.append(last_entropy)
        avg_entropy = np.mean(entropies[-10:])
        avg_entropies.append(avg_entropy)

        if episode % 100 == 0:
            # 100 episode마다 terminal과 csv에 학습 상태 기록
            print(
                f"Ep {episode} | Avg Rew: {avg_reward:.1f} | BEST Rew: {best_reward:.1f} "
                f"| A_Loss: {last_actor_loss:.3f} | V_Loss: {last_value_loss:.3f} "
                f"| Entropy: {last_entropy:.3f}"
            )
            with open(log_path, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        episode,
                        f"{avg_reward:.1f}",
                        f"{best_reward:.1f}",
                        f"{last_actor_loss:.3f}",
                        f"{last_value_loss:.3f}",
                        f"{last_entropy:.3f}",
                    ]
                )

        # moving average reward가 best를 갱신하면 checkpoint 저장
        if avg_reward > best_reward:
            best_reward = avg_reward
            no_improve_count = 0
            save_model(model, optimizer, episode, best_reward, model_path)
        else:
            no_improve_count += 1

        # target reward를 넘긴 상태에서 오래 개선이 없으면 학습 종료
        if no_improve_count >= patience and avg_reward >= target_reward:
            print(f"Early Stopping! Target reached at episode {episode}")
            break

    env.close()

    # 학습이 끝나면 reward, actor loss, value loss, entropy 그래프를 저장
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].plot(all_rewards, alpha=0.3, color="blue")
    axs[0].plot(avg_rewards, color="red", label="Moving Avg (10)")
    axs[0].axhline(y=target_reward, color="green", linestyle="--")
    axs[0].set_title("Episode Reward")
    axs[0].legend()

    axs[1].plot(actor_losses, color="orange")
    axs[1].plot(avg_actor_losses, color="red", label="Moving Avg (10)")
    axs[1].set_title("Actor Loss (PPO Clipped)")
    axs[1].legend()

    axs[2].plot(value_losses, color="purple")
    axs[2].plot(avg_value_losses, color="red", label="Moving Avg (10)")
    axs[2].set_title("Value Loss (MSE)")
    axs[2].legend()

    axs[3].plot(entropies, color="teal")
    axs[3].plot(avg_entropies, color="red", label="Moving Avg (10)")
    axs[3].set_title("Policy Entropy")
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Metrics plot saved as '{plot_path}'")
    plt.show()


if __name__ == "__main__":
    train()
