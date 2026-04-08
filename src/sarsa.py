import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


# -------------------------------------------------
# basic setting
# -------------------------------------------------
state_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
action_space = [0, 1]

goal_prefix = [0, 1, 0, 1, 0]
goal_sequence = [0, 1, 0, 1, 0, 0]

alpha = 0.1
gamma = 0.99

epsilon = 0.3
epsilon_decay = 0.995
epsilon_min = 0.01

num_episodes = 5000


# -------------------------------------------------
# environment-related functions
# -------------------------------------------------
def terminal(pos):
    return pos == len(state_names) - 1


def state_to_key(pos, history):
    return (pos, tuple(history))


def reward_function(pos, action, history):
    current_state = state_names[pos]

    if action == 1:
        return 1

    if current_state == 'x6' and list(history) == goal_prefix:
        return 1000

    return 0


# -------------------------------------------------
# Q storage
# -------------------------------------------------
Q = defaultdict(lambda: torch.zeros(len(action_space), dtype=torch.float32))


def prepare_state(key):
    if key not in Q:
        Q[key] = torch.ones(len(action_space), dtype=torch.float32) * 5.0


def epsilon_greedy(key, eps):
    prepare_state(key)

    if np.random.rand() < eps:
        return np.random.choice(action_space)

    return torch.argmax(Q[key]).item()


# -------------------------------------------------
# logs
# -------------------------------------------------
episode_rewards = []
episode_td_means = []
episode_value_means = []
success_history = []


# -------------------------------------------------
# training
# -------------------------------------------------
for episode in range(num_episodes):
    pos = 0
    action_history = []

    current_key = state_to_key(pos, action_history)
    current_action = epsilon_greedy(current_key, epsilon)

    total_reward = 0.0
    td_list = []
    visited = set()
    success = 0

    while True:
        visited.add(current_key)

        reward = reward_function(pos, current_action, action_history)
        total_reward += reward

        next_history = action_history + [current_action]

        if terminal(pos):
            target = torch.tensor(float(reward))
            error = target - Q[current_key][current_action]
            Q[current_key][current_action] += alpha * error
            td_list.append(error.item())

            if next_history == goal_sequence:
                success = 1
            break

        next_pos = pos + 1
        next_key = state_to_key(next_pos, next_history)
        next_action = epsilon_greedy(next_key, epsilon)

        prepare_state(next_key)

        # SARSA update
        target = reward + gamma * Q[next_key][next_action]
        error = target - Q[current_key][current_action]
        Q[current_key][current_action] += alpha * error
        td_list.append(error.item())

        pos = next_pos
        action_history = next_history
        current_key = next_key
        current_action = next_action

    episode_rewards.append(total_reward)
    episode_td_means.append(float(np.mean(td_list)))
    episode_value_means.append(np.mean([torch.max(Q[k]).item() for k in visited]))
    success_history.append(success)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 200 == 0 or episode == num_episodes - 1:
        print(
            f"[SARSA] episode {episode:4d} | "
            f"reward {total_reward:7.1f} | "
            f"td {episode_td_means[-1]:8.4f} | "
            f"value {episode_value_means[-1]:8.2f} | "
            f"success {sum(success_history):4d} | "
            f"eps {epsilon:.4f}"
        )


# -------------------------------------------------
# check learned greedy sequence
# -------------------------------------------------
def run_greedy_policy():
    pos = 0
    history = []
    chosen = []
    total = 0.0

    while True:
        key = state_to_key(pos, history)
        prepare_state(key)

        action = torch.argmax(Q[key]).item()
        chosen.append(action)

        total += reward_function(pos, action, history)
        history = history + [action]

        if terminal(pos):
            break

        pos += 1

    return chosen, total


best_sequence, best_reward = run_greedy_policy()

print("\n=== SARSA result ===")
print("target sequence :", goal_sequence)
print("learned sequence:", best_sequence)
print("greedy reward   :", best_reward)
print("success count   :", sum(success_history))


# -------------------------------------------------
# visualization
# -------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(episode_rewards)
plt.title("SARSA - Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(episode_value_means)
plt.title("SARSA - Mean Value")
plt.xlabel("Episode")
plt.ylabel("Mean max Q")
plt.grid(True)
plt.show()