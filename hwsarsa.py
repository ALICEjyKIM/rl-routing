import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    import torch
except ModuleNotFoundError:
    print("torch가 설치된 Python으로 실행해야 합니다.")
    print("현재 Python:", sys.executable)
    raise

print("Python:", sys.executable)
print("CWD   :", os.getcwd())
print("Torch :", torch.__version__)

# =========================================================
# 1. Problem setting
# =========================================================
states = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
actions = [0, 1]
target_prefix = [0, 1, 0, 1, 0]          # x6 직전까지의 action history
target_full_policy = [0, 1, 0, 1, 0, 0]  # 전체 목표 정책

alpha = 0.1
gamma = 0.99
epsilon = 0.3
epsilon_decay = 0.995
epsilon_min = 0.01
epochs = 5000

# =========================================================
# 2. Augmented state
#    state = (position_index, action_history_tuple)
# =========================================================
def get_reward(pos, action, history):
    curr_state = states[pos]

    if action == 1:
        return 1
    elif action == 0:
        if curr_state == 'x6' and list(history) == target_prefix:
            return 1000
        else:
            return 0

def is_terminal(pos):
    return pos == len(states) - 1  # x6에서 action 수행 후 종료

def make_key(pos, history):
    return (pos, tuple(history))

# Q-table: dict of tensors, each state has 2 action values
Q = defaultdict(lambda: torch.zeros(len(actions), dtype=torch.float32))

# optimistic init helps exploration a bit
def ensure_state_exists(key):
    if key not in Q:
        Q[key] = torch.ones(len(actions), dtype=torch.float32) * 5.0

def choose_action(state_key, epsilon):
    ensure_state_exists(state_key)
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    return torch.argmax(Q[state_key]).item()

# =========================================================
# 3. Logs
# =========================================================
reward_log = []
value_mean_log = []
td_error_log = []
success_log = []

# =========================================================
# 4. SARSA training
# =========================================================
for epoch in range(epochs):
    pos = 0
    history = []

    state_key = make_key(pos, history)
    action = choose_action(state_key, epsilon)

    total_reward = 0.0
    episode_td_errors = []
    success = 0
    visited_keys = set()

    while True:
        visited_keys.add(state_key)

        reward = get_reward(pos, action, history)
        total_reward += reward

        next_history = history + [action]

        # terminal: acted at x6, then episode ends
        if is_terminal(pos):
            td_target = torch.tensor(float(reward))
            td_error = td_target - Q[state_key][action]
            Q[state_key][action] += alpha * td_error
            episode_td_errors.append(td_error.item())

            if next_history == target_full_policy:
                success = 1
            break

        next_pos = pos + 1
        next_state_key = make_key(next_pos, next_history)
        next_action = choose_action(next_state_key, epsilon)

        ensure_state_exists(next_state_key)

        td_target = reward + gamma * Q[next_state_key][next_action]
        td_error = td_target - Q[state_key][action]
        Q[state_key][action] += alpha * td_error
        episode_td_errors.append(td_error.item())

        pos = next_pos
        history = next_history
        state_key = next_state_key
        action = next_action

    reward_log.append(total_reward)
    td_error_log.append(float(np.mean(episode_td_errors)))
    success_log.append(success)

    # mean V(s) over visited augmented states in this episode
    v_vals = [torch.max(Q[k]).item() for k in visited_keys]
    value_mean_log.append(float(np.mean(v_vals)))

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if epoch % 200 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch:4d} | "
            f"Reward={total_reward:8.2f} | "
            f"TD={td_error_log[-1]:8.4f} | "
            f"V_mean={value_mean_log[-1]:8.2f} | "
            f"Success={sum(success_log):4d} | "
            f"eps={epsilon:.4f}"
        )

# =========================================================
# 5. Greedy rollout to inspect learned policy
# =========================================================
def greedy_rollout():
    pos = 0
    history = []
    chosen_actions = []
    total_reward = 0.0

    while True:
        key = make_key(pos, history)
        ensure_state_exists(key)
        action = torch.argmax(Q[key]).item()
        chosen_actions.append(action)

        reward = get_reward(pos, action, history)
        total_reward += reward

        history = history + [action]

        if is_terminal(pos):
            break
        pos += 1

    return chosen_actions, total_reward

learned_policy, greedy_reward = greedy_rollout()

# =========================================================
# 6. Print results
# =========================================================
print("\n=== Greedy Rollout Policy ===")
print("Learned Policy:", learned_policy)
print("Greedy Rollout Reward:", greedy_reward)
print("Target Policy:", target_full_policy)

print("\n=== State Value Function on Greedy Path ===")
pos = 0
history = []
while True:
    key = make_key(pos, history)
    ensure_state_exists(key)
    v = torch.max(Q[key]).item()
    print(f"state=({states[pos]}, history={history}) -> V={v:.2f}")

    action = torch.argmax(Q[key]).item()
    history = history + [action]
    if is_terminal(pos):
        break
    pos += 1

print("\n=== Learned Q-values on Greedy Path ===")
pos = 0
history = []
while True:
    key = make_key(pos, history)
    ensure_state_exists(key)
    print(f"state=({states[pos]}, history={history}) | Q(0)={Q[key][0].item():.2f}, Q(1)={Q[key][1].item():.2f}")

    action = torch.argmax(Q[key]).item()
    history = history + [action]
    if is_terminal(pos):
        break
    pos += 1

# =========================================================
# 7. Plots
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(reward_log)
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(value_mean_log)
plt.title("Value Function Mean (visited augmented states)")
plt.xlabel("Episode")
plt.ylabel("Mean V")
plt.grid(True)
plt.show()