import csv
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

import torch.serialization

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.serialization.add_safe_globals([np._core.multiarray.scalar])

# 하이퍼파라미터
gamma = 0.99
lr = 0.0003
patience = 50  # 성능 개선이 없을 때 기다릴 에피소드 수
target_reward = 200  # LunarLander 해결 기준 (보통 200점 이상)

log_path = os.path.join(os.path.dirname(__file__), "A2C_training_log.csv")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.mu = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std)
        value = self.value(x)
        return mu, std, value

def save_model(model, optimizer, episode, best_reward, path):
    """
    모델 가중치뿐만 아니라 학습 상태 전체를 저장
    """
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_reward': best_reward
    }
    torch.save(checkpoint, path)
    print(f"--- Checkpoint saved at episode {episode} (Best: {best_reward:.2f}) ---")

def train():
    env = gym.make("LunarLanderContinuous-v3")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["episode", "avg_reward", "best_reward", "actor_loss", "value_loss"])

    all_rewards, avg_rewards = [], []
    actor_losses, value_losses, avg_actor_losses, avg_value_losses = [], [], [], []
    best_reward = -float('inf')
    no_improve_count = 0

    for episode in range(10000):
        state, _ = env.reset()
        log_probs, values, rewards, masks = [], [], [], []
        entropy = 0

        for t in range(1000):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mu, std, value = model(state_tensor)
            
            dist = Normal(mu, std)
            action = dist.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
            done = terminated or truncated

            log_probs.append(dist.log_prob(action).sum(dim=-1))
            values.append(value)
            rewards.append(torch.FloatTensor([reward]))
            masks.append(torch.FloatTensor([1 - done]))

            state = next_state
            if done: break

        # Returns & Advantage 계산
        next_v = model(torch.FloatTensor(next_state).unsqueeze(0))[2]
        returns = []
        R = next_v.detach()
        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + gamma * R * m
            returns.insert(0, R)
            
        returns = torch.stack(returns).squeeze()
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs).squeeze()
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 결과 기록
        episode_reward = sum(rewards).item()
        all_rewards.append(episode_reward)
        avg_reward = np.mean(all_rewards[-10:])
        avg_rewards.append(avg_reward)
        
        actor_losses.append(actor_loss.item())
        avg_actor_loss = np.mean(actor_losses[-10:])
        avg_actor_losses.append(avg_actor_loss)

        value_losses.append(critic_loss.item())
        avg_value_loss = np.mean(value_losses[-10:])
        avg_value_losses.append(avg_value_loss)

        if episode % 100 == 0:
            print(f"Ep {episode} | Avg Rew: {avg_reward:.1f} | BEST Rew: {best_reward:.1f} | A_Loss: {actor_loss.item():.3f} | V_Loss: {critic_loss.item():.3f}")
            with open(log_path, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    episode,
                    f"{avg_reward:.1f}",
                    f"{best_reward:.1f}",
                    f"{actor_loss.item():.3f}",
                    f"{critic_loss.item():.3f}",
                ])

        # --- Early Stopping Logic ---
        if avg_reward > best_reward:
            best_reward = avg_reward
            no_improve_count = 0
            save_model(model, optimizer, episode, best_reward, "A2C_basic_best.pth")
        else:
            no_improve_count += 1

        if no_improve_count >= patience and avg_reward >= target_reward:
            print(f"Early Stopping! Target reached at episode {episode}")
            break

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Reward
    axs[0].plot(all_rewards, alpha=0.3, color='blue')
    axs[0].plot(avg_rewards, color='red', label='Moving Avg (10)')
    axs[0].axhline(y=target_reward, color='green', linestyle='--')
    axs[0].set_title('Episode Reward')
    axs[0].legend()

    # 2. Actor Loss
    axs[1].plot(actor_losses, color='orange')
    axs[1].plot(avg_actor_losses, color='red', label='Moving Avg (10)')
    axs[1].set_title('Actor Loss (Policy Gradient)')

    # 3. Value Loss
    axs[2].plot(value_losses, color='purple')
    axs[2].plot(avg_value_losses, color='red', label='Moving Avg (10)')
    axs[2].set_title('Value Loss (MSE)')

    plt.tight_layout()
    plt.savefig('A2C_basic_metrics.png')
    print("Metrics plot saved as 'A2C_basic_metrics.png'")
    plt.show()

if __name__ == "__main__":
    train()
