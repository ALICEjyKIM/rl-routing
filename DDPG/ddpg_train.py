import csv
import os
import random
import warnings
from collections import deque

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.serialization

torch.serialization.add_safe_globals([np._core.multiarray.scalar])


gamma = 0.99
actor_lr = 0.0003
critic_lr = 0.001
tau = 0.005
batch_size = 128
buffer_size = 100000
warmup_steps = 5000
max_steps = 1000
max_episodes = 10000
patience = 50
target_reward = 200
noise_std_start = 0.2
noise_std_min = 0.05
noise_decay = 0.995

base_dir = os.path.dirname(__file__)
log_path = os.path.join(base_dir, "DDPG_training_log.csv")
model_path = os.path.join(base_dir, "DDPG_basic_best.pth")
plot_path = os.path.join(base_dir, "DDPG_basic_metrics.png")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)

    def forward(self, state):
        return self.net(state) * self.action_scale + self.action_bias


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, size, device):
        batch = random.sample(self.buffer, size)
        states, actions, rewards, next_states, terminals = map(np.array, zip(*batch))

        return (
            torch.as_tensor(states, dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.float32, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(next_states, dtype=torch.float32, device=device),
            torch.as_tensor(terminals, dtype=torch.float32, device=device).unsqueeze(1),
        )


def soft_update(source, target):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


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
                    "critic_loss",
                    "noise_std",
                ]
            )


def save_model(actor, critic, actor_optimizer, critic_optimizer, episode, best_reward, path):
    checkpoint = {
        "episode": episode,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "best_reward": best_reward,
    }
    torch.save(checkpoint, path)
    print(f"--- Checkpoint saved at episode {episode} (Best: {best_reward:.2f}) ---")


def train():
    env = gym.make("LunarLanderContinuous-v3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low_np = env.action_space.low
    action_high_np = env.action_space.high
    action_low = torch.as_tensor(action_low_np, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(action_high_np, dtype=torch.float32, device=device)

    actor = Actor(state_dim, action_dim, action_low_np, action_high_np).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, action_dim, action_low_np, action_high_np).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)
    hard_update(actor, target_actor)
    hard_update(critic, target_critic)

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    replay_buffer = ReplayBuffer(buffer_size)

    make_log_file()

    all_rewards, avg_rewards = [], []
    actor_losses, critic_losses = [], []
    avg_actor_losses, avg_critic_losses = [], []
    noise_stds, avg_noise_stds = [], []

    best_reward = -float("inf")
    no_improve_count = 0
    total_steps = 0
    noise_std = noise_std_start

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        last_actor_loss = 0.0
        last_critic_loss = 0.0

        for _ in range(max_steps):
            if total_steps < warmup_steps:
                env_action = env.action_space.sample()
            else:
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = actor(state_tensor).squeeze(0)
                noise = torch.normal(mean=0.0, std=noise_std, size=action.shape, device=device)
                action = torch.clamp(action + noise, action_low, action_high)
                env_action = action.cpu().numpy()

            # Gymnasium separates natural terminal states from time-limit truncation.
            # Stop the episode on either one, but only mask bootstrapping on terminated.
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            episode_done = terminated or truncated
            replay_buffer.push(state, env_action, reward, next_state, terminated)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, terminals = replay_buffer.sample(batch_size, device)

                with torch.no_grad():
                    next_actions = target_actor(next_states)
                    target_q = target_critic(next_states, next_actions)
                    # If an episode was only truncated by a time limit, keep bootstrapping
                    # from next_state. True terminal states should stop the target.
                    q_target = rewards + gamma * (1.0 - terminals) * target_q

                q_value = critic(states, actions)
                critic_loss = F.mse_loss(q_value, q_target)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = -critic(states, actor(states)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                soft_update(actor, target_actor)
                soft_update(critic, target_critic)

                last_actor_loss = actor_loss.item()
                last_critic_loss = critic_loss.item()

            if episode_done:
                break

        noise_std = max(noise_std_min, noise_std * noise_decay)

        all_rewards.append(episode_reward)
        avg_reward = np.mean(all_rewards[-10:])
        avg_rewards.append(avg_reward)

        actor_losses.append(last_actor_loss)
        avg_actor_loss = np.mean(actor_losses[-10:])
        avg_actor_losses.append(avg_actor_loss)

        critic_losses.append(last_critic_loss)
        avg_critic_loss = np.mean(critic_losses[-10:])
        avg_critic_losses.append(avg_critic_loss)

        noise_stds.append(noise_std)
        avg_noise_std = np.mean(noise_stds[-10:])
        avg_noise_stds.append(avg_noise_std)

        if episode % 100 == 0:
            print(
                f"Ep {episode} | Avg Rew: {avg_reward:.1f} | BEST Rew: {best_reward:.1f} "
                f"| A_Loss: {last_actor_loss:.3f} | C_Loss: {last_critic_loss:.3f} "
                f"| Noise: {noise_std:.3f}"
            )
            with open(log_path, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        episode,
                        f"{avg_reward:.1f}",
                        f"{best_reward:.1f}",
                        f"{last_actor_loss:.3f}",
                        f"{last_critic_loss:.3f}",
                        f"{noise_std:.3f}",
                    ]
                )

        if avg_reward > best_reward:
            best_reward = avg_reward
            no_improve_count = 0
            save_model(actor, critic, actor_optimizer, critic_optimizer, episode, best_reward, model_path)
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
    axs[1].set_title("Actor Loss")
    axs[1].legend()

    axs[2].plot(critic_losses, color="purple")
    axs[2].plot(avg_critic_losses, color="red", label="Moving Avg (10)")
    axs[2].set_title("Critic Loss (MSE)")
    axs[2].legend()

    axs[3].plot(noise_stds, color="teal")
    axs[3].plot(avg_noise_stds, color="red", label="Moving Avg (10)")
    axs[3].set_title("Exploration Noise")
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Metrics plot saved as '{plot_path}'")
    plt.show()


if __name__ == "__main__":
    train()
