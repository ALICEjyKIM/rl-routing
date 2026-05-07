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
gamma = 0.99
lr = 0.0003
clip_epsilon = 0.2
ppo_epochs = 10
entropy_coef = 0.01
value_coef = 0.5
max_steps = 1000
max_episodes = 10000
patience = 50
target_reward = 200

base_dir = os.path.dirname(__file__)
log_path = os.path.join(base_dir, "PPO_training_log.csv")
model_path = os.path.join(base_dir, "PPO_basic_best.pth")
plot_path = os.path.join(base_dir, "PPO_basic_metrics.png")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )

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
    checkpoint = {
        "episode": episode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_reward": best_reward,
    }
    torch.save(checkpoint, path)
    print(f"--- Checkpoint saved at episode {episode} (Best: {best_reward:.2f}) ---")


def make_log_file():
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
    returns = []
    R = last_value.detach()

    for reward, mask in zip(reversed(rewards), reversed(masks)):
        R = reward + gamma * R * mask
        returns.insert(0, R)

    return torch.stack(returns).view(-1).to(device)


def train():
    env = gym.make("LunarLanderContinuous-v3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)

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
        state, _ = env.reset()

        states, actions, old_log_probs = [], [], []
        rewards, masks, values = [], [], []
        episode_reward = 0.0

        for _ in range(max_steps):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                mu, std, value = model(state_tensor)
                dist = Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            env_action = torch.clamp(action.squeeze(0), action_low, action_high)
            next_state, reward, terminated, truncated, _ = env.step(env_action.cpu().numpy())
            done = terminated or truncated

            states.append(state_tensor.squeeze(0))
            actions.append(action.squeeze(0))
            old_log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))
            masks.append(torch.tensor([1 - done], dtype=torch.float32, device=device))

            episode_reward += reward
            state = next_state

            if done:
                break

        with torch.no_grad():
            next_state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = model(next_state_tensor)[2]

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs).detach()
        old_values = torch.stack(values).view(-1).detach()
        returns = compute_returns(rewards, masks, last_value, device)

        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        last_actor_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0

        for _ in range(ppo_epochs):
            mu, std, new_values = model(states)
            dist = Normal(mu, std)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratios = torch.exp(new_log_probs - old_log_probs)
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = F.mse_loss(new_values.view(-1), returns)
            loss = actor_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_actor_loss = actor_loss.item()
            last_value_loss = value_loss.item()
            last_entropy = entropy.item()

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

        if avg_reward > best_reward:
            best_reward = avg_reward
            no_improve_count = 0
            save_model(model, optimizer, episode, best_reward, model_path)
        else:
            no_improve_count += 1

        if no_improve_count >= patience and avg_reward >= target_reward:
            print(f"Early Stopping! Target reached at episode {episode}")
            break

    env.close()

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
