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
from torch.distributions import Normal

import torch.serialization

torch.serialization.add_safe_globals([np._core.multiarray.scalar])


gamma = 0.99
actor_lr = 0.0003
critic_lr = 0.001
alpha_lr = 0.0003
tau = 0.005
batch_size = 128
buffer_size = 100000
warmup_steps = 5000
max_steps = 1000
max_episodes = 10000
patience = 50
target_reward = 200
log_std_min = -20
log_std_max = 2
epsilon = 1e-6

base_dir = os.path.dirname(__file__)
log_path = os.path.join(base_dir, "SAC_training_log.csv")
model_path = os.path.join(base_dir, "SAC_basic_best.pth")
plot_path = os.path.join(base_dir, "SAC_basic_metrics.png")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super(Actor, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)

    def forward(self, state):
        hidden = self.backbone(state)
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden)
        log_std = torch.clamp(log_std, log_std_min, log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_scale + self.action_bias

        log_prob = normal.log_prob(raw_action)
        correction = torch.log(self.action_scale * (1.0 - squashed_action.pow(2)) + epsilon)
        log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)

        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.q1(state_action), self.q2(state_action)


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
                    "alpha",
                ]
            )


def save_model(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    log_alpha,
    alpha_optimizer,
    episode,
    best_reward,
    path,
):
    checkpoint = {
        "episode": episode,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
        "best_reward": best_reward,
        "algorithm": "SAC",
        "paper_features": {
            "maximum_entropy_objective": True,
            "reparameterized_tanh_gaussian_policy": True,
            "clipped_double_q_learning": True,
            "automatic_entropy_temperature": True,
        },
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

    actor = Actor(state_dim, action_dim, action_low_np, action_high_np).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)
    hard_update(critic, target_critic)

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    target_entropy = -float(action_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=alpha_lr)

    replay_buffer = ReplayBuffer(buffer_size)
    make_log_file()

    all_rewards, avg_rewards = [], []
    actor_losses, critic_losses = [], []
    avg_actor_losses, avg_critic_losses = [], []
    alphas, avg_alphas = [], []

    best_reward = -float("inf")
    no_improve_count = 0
    total_steps = 0

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
                    action, _, _ = actor.sample(state_tensor)
                env_action = action.squeeze(0).cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(env_action)
            episode_done = terminated or truncated
            replay_buffer.push(state, env_action, reward, next_state, terminated)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, terminals = replay_buffer.sample(batch_size, device)
                alpha = log_alpha.exp()

                with torch.no_grad():
                    next_actions, next_log_probs, _ = actor.sample(next_states)
                    target_q1, target_q2 = target_critic(next_states, next_actions)
                    target_q = torch.minimum(target_q1, target_q2) - alpha * next_log_probs
                    q_target = rewards + gamma * (1.0 - terminals) * target_q

                current_q1, current_q2 = critic(states, actions)
                critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                new_actions, log_probs, _ = actor.sample(states)
                q1_new, q2_new = critic(states, new_actions)
                q_new = torch.minimum(q1_new, q2_new)
                actor_loss = (alpha.detach() * log_probs - q_new).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()

                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                soft_update(critic, target_critic)

                last_actor_loss = actor_loss.item()
                last_critic_loss = critic_loss.item()

            if episode_done:
                break

        current_alpha = log_alpha.exp().item()

        all_rewards.append(episode_reward)
        avg_reward = np.mean(all_rewards[-10:])
        avg_rewards.append(avg_reward)

        actor_losses.append(last_actor_loss)
        avg_actor_loss = np.mean(actor_losses[-10:])
        avg_actor_losses.append(avg_actor_loss)

        critic_losses.append(last_critic_loss)
        avg_critic_loss = np.mean(critic_losses[-10:])
        avg_critic_losses.append(avg_critic_loss)

        alphas.append(current_alpha)
        avg_alpha = np.mean(alphas[-10:])
        avg_alphas.append(avg_alpha)

        if episode % 100 == 0:
            print(
                f"Ep {episode} | Avg Rew: {avg_reward:.1f} | BEST Rew: {best_reward:.1f} "
                f"| A_Loss: {last_actor_loss:.3f} | C_Loss: {last_critic_loss:.3f} "
                f"| Alpha: {current_alpha:.3f}"
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
                        f"{current_alpha:.3f}",
                    ]
                )

        if avg_reward > best_reward:
            best_reward = avg_reward
            no_improve_count = 0
            save_model(
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                log_alpha,
                alpha_optimizer,
                episode,
                best_reward,
                model_path,
            )
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
    axs[2].set_title("Twin Critic Loss (MSE)")
    axs[2].legend()

    axs[3].plot(alphas, color="teal")
    axs[3].plot(avg_alphas, color="red", label="Moving Avg (10)")
    axs[3].set_title("Entropy Temperature")
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Metrics plot saved as '{plot_path}'")
    plt.show()


if __name__ == "__main__":
    train()
