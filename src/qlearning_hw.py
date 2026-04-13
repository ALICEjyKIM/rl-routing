import numpy as np
import torch
import matplotlib.pyplot as plt

# 1. Environment setting
states = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
actions = [0, 1]
target_policy = [0, 1, 0, 1, 0, 0]   # 원하는 최종 정책

num_states = len(states)
num_actions = len(actions)

def get_next_state_idx(state_idx):
    if state_idx < num_states - 1:
        return state_idx + 1
    return None  # x6 이후 terminal

def get_reward(state_idx, action, action_history):
    """
    Reward shaping:
    1) 현재 state에서 target action과 맞으면 +10
    2) 틀리면 -10
    3) 마지막 state(x6)까지 전체 sequence를 정확히 맞추면 +1000 추가
    """
    reward = 0

    # 현재 상태에서 target action과 일치 여부
    if action == target_policy[state_idx]:
        reward += 10
    else:
        reward -= 10

    # 마지막 상태에서 전체 시퀀스 일치 여부 확인
    if state_idx == num_states - 1:
        full_sequence = action_history + [action]
        if full_sequence == target_policy:
            reward += 1000

    return reward

# 2. Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.3
epsilon_min = 0.01
epsilon_decay = 0.995
epochs = 3000

# 초기 Q-table
q_table = torch.zeros((num_states, num_actions), dtype=torch.float32)

# 3. Policy
def choose_action(state_idx, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return torch.argmax(q_table[state_idx]).item()

# 4. Logs
reward_log = []
value_mean_log = []
td_error_log = []
success_log = []

# 5. Training loop (Q-learning)
for epoch in range(epochs):
    state_idx = 0
    done = False
    total_reward = 0.0
    action_history = []
    episode_td_errors = []
    success = 0

    while not done:
        action = choose_action(state_idx, epsilon)
        reward = get_reward(state_idx, action, action_history)
        total_reward += reward

        next_state_idx = get_next_state_idx(state_idx)

        if next_state_idx is None:
            # terminal update
            td_target = torch.tensor(float(reward))
            done = True

            # 성공 여부 기록
            if action_history + [action] == target_policy:
                success = 1
        else:
            max_next_q = torch.max(q_table[next_state_idx]).item()
            td_target = torch.tensor(float(reward + gamma * max_next_q))

        td_error = td_target - q_table[state_idx, action]
        q_table[state_idx, action] += alpha * td_error

        episode_td_errors.append(td_error.item())

        action_history.append(action)

        if next_state_idx is not None:
            state_idx = next_state_idx

    reward_log.append(total_reward)
    value_mean_log.append(torch.max(q_table, dim=1).values.mean().item())
    td_error_log.append(np.mean(episode_td_errors))
    success_log.append(success)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch:4d} | "
            f"Reward = {total_reward:8.2f} | "
            f"Mean TD Error = {np.mean(episode_td_errors):8.4f} | "
            f"V_mean = {value_mean_log[-1]:8.2f} | "
            f"Success Count = {sum(success_log):4d} | "
            f"Epsilon = {epsilon:6.4f}"
        )

# 6. Results
print("\nLearned Q-Table")
print("State | Action 0 | Action 1")
for i, s in enumerate(states):
    print(f"{s:5} | {q_table[i, 0].item():10.2f} | {q_table[i, 1].item():10.2f}")

print("\nState Value Function V(s) = max_a Q(s,a)")
state_values = torch.max(q_table, dim=1).values
for i, s in enumerate(states):
    print(f"{s}: {state_values[i].item():.2f}")

print("\nLearned Policy")
learned_policy = []
for i, s in enumerate(states):
    best_action = torch.argmax(q_table[i]).item()
    learned_policy.append(best_action)
    print(f"{s} -> Action {best_action}")

print("\nTarget Policy :", target_policy)
print("Learned Policy:", learned_policy)

# 7. Plots
plt.figure(figsize=(8, 5))
plt.plot(reward_log)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(value_mean_log)
plt.xlabel("Episode")
plt.ylabel("Mean of V(s)")
plt.title("Value Function Mean")
plt.grid(True)
plt.show()