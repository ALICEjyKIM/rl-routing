import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. Problem setting
# =========================================================
states = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
actions = [0, 1]
target_sequence = [0, 1, 0, 1, 0]   # reward condition before acting at x6

num_states = len(states)
num_actions = len(actions)

def get_reward(state, action, prev_actions):
    # Problem statement:
    # if action == 1: return 1
    # elif action == 0:
    #   if current_state == 'x6' and prev_actions == [0,1,0,1,0]:
    #       return 1000
    #   else:
    #       return 0
    if action == 1:
        return 1
    elif action == 0:
        if state == 'x6' and prev_actions == target_sequence:
            return 1000
        else:
            return 0

# =========================================================
# 2. Hyperparameters
# =========================================================
alpha = 0.08
gamma = 0.99

# exploration: start high, decay slowly
epsilon = 0.80
epsilon_decay = 0.9995
epsilon_min = 0.05

# enough training
epochs = 20000

# optimistic initialization to encourage exploration
q_table = np.ones((num_states, num_actions), dtype=np.float64) * 5.0

# =========================================================
# 3. Policy
# =========================================================
def choose_action(state_idx, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    return int(np.argmax(q_table[state_idx]))

# =========================================================
# 4. Logs
# =========================================================
reward_log = []
value_mean_log = []
td_error_log = []
success_log = []

# =========================================================
# 5. SARSA training (general SARSA: state = x1~x6 only)
# =========================================================
for epoch in range(epochs):
    state_idx = 0
    prev_actions = []
    total_reward = 0.0
    episode_td_errors = []
    success = 0

    action = choose_action(state_idx, epsilon)

    while True:
        curr_state = states[state_idx]
        reward = get_reward(curr_state, action, prev_actions)
        total_reward += reward

        # terminal update at x6
        if state_idx == num_states - 1:
            td_target = reward
            td_error = td_target - q_table[state_idx, action]
            q_table[state_idx, action] += alpha * td_error
            episode_td_errors.append(td_error)

            if prev_actions + [action] == [0, 1, 0, 1, 0, 0]:
                success = 1
            break

        next_state_idx = state_idx + 1
        next_action = choose_action(next_state_idx, epsilon)

        td_target = reward + gamma * q_table[next_state_idx, next_action]
        td_error = td_target - q_table[state_idx, action]
        q_table[state_idx, action] += alpha * td_error
        episode_td_errors.append(td_error)

        prev_actions.append(action)
        state_idx = next_state_idx
        action = next_action

    reward_log.append(total_reward)
    value_mean_log.append(np.mean(np.max(q_table, axis=1)))
    td_error_log.append(np.mean(episode_td_errors))
    success_log.append(success)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if epoch % 1000 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch:5d} | "
            f"Reward = {total_reward:8.2f} | "
            f"Mean TD Error = {td_error_log[-1]:9.4f} | "
            f"V_mean = {value_mean_log[-1]:8.2f} | "
            f"Success Count = {sum(success_log):5d} | "
            f"Epsilon = {epsilon:6.4f}"
        )

# =========================================================
# 6. Results
# =========================================================
print("\n" + "=" * 60)
print("Learned Q-table")
print("=" * 60)
print("State | Action 0 | Action 1")
for i, s in enumerate(states):
    print(f"{s:5} | {q_table[i, 0]:10.2f} | {q_table[i, 1]:10.2f}")

print("\n" + "=" * 60)
print("State Value Function V(s) = max_a Q(s,a)")
print("=" * 60)
state_values = np.max(q_table, axis=1)
for i, s in enumerate(states):
    print(f"{s}: {state_values[i]:.2f}")

print("\n" + "=" * 60)
print("Learned Policy")
print("=" * 60)
learned_policy = []
for i, s in enumerate(states):
    best_action = int(np.argmax(q_table[i]))
    learned_policy.append(best_action)
    print(f"{s} -> Action {best_action}")

print("\nTarget sequence for full success: [0, 1, 0, 1, 0, 0]")
print("Learned policy:", learned_policy)
print("Number of successful episodes:", sum(success_log))

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
plt.title("Value Function Mean")
plt.xlabel("Episode")
plt.ylabel("Mean of V(s)")
plt.grid(True)
plt.show()