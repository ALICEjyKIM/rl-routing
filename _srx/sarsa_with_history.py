import os
import csv
import random
from datetime import datetime

import torch
import matplotlib.pyplot as plt


class ChainEnvWithHistory:
    """
    히스토리를 상태에 포함한 환경

    원래 physical state:
        x1 -> x2 -> x3 -> x4 -> x5 -> x6

    action:
        0, 1

    reward:
        if action == 1:
            reward = 1
        elif action == 0:
            if current_state == x6 and prev_actions == [0, 1, 0, 1, 0]:
                reward = 1000
            else:
                reward = 0

    핵심:
    - 같은 x6라도, 어떤 action history로 왔는지에 따라
      최종 보상이 달라지므로
      상태를 (현재 위치, 이전 action들)로 확장해야 함.
    - 이렇게 해야 마르코프하게 됨.
    """

    def __init__(self):
        self.num_actions = 2
        self.num_positions = 6   # x1 ~ x6
        self.reset()

    def reset(self):
        self.position = 0          # x1 -> 0
        self.prev_actions = []     # x1~x5에서 취한 action 저장
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        상태를 튜플로 반환:
        (position, history_tuple)

        예:
        x1, history=[]           -> (0, ())
        x3, history=[0,1]        -> (2, (0,1))
        x6, history=[0,1,0,1,0]  -> (5, (0,1,0,1,0))
        """
        return (self.position, tuple(self.prev_actions))

    def step(self, action):
        if self.done:
            raise ValueError("Episode is done. Call reset().")

        current_position = self.position
        reward = 0

        # 마지막 상태 x6 에서의 행동
        if current_position == 5:
            if action == 1:
                reward = 1
            else:
                if self.prev_actions == [0, 1, 0, 1, 0]:
                    reward = 1000
                else:
                    reward = 0

            self.done = True
            next_state = self.get_state()
            return next_state, reward, self.done

        # x1 ~ x5 에서의 행동
        if action == 1:
            reward = 1
        else:
            reward = 0

        self.prev_actions.append(action)
        self.position += 1
        next_state = self.get_state()

        return next_state, reward, self.done


def epsilon_greedy(Q, state, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)

    if state not in Q:
        Q[state] = torch.zeros(num_actions, dtype=torch.float32)

    return int(torch.argmax(Q[state]).item())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def moving_average(values, window):
    if len(values) < window:
        return []
    result = []
    for i in range(len(values) - window + 1):
        result.append(sum(values[i:i + window]) / window)
    return result


def estimate_convergence(episode_rewards, window=100, tolerance=1e-3):
    if len(episode_rewards) < 2 * window:
        return None

    ma = moving_average(episode_rewards, window)
    for i in range(1, len(ma)):
        if abs(ma[i] - ma[i - 1]) < tolerance:
            return i + window
    return None


def get_q_row(Q, state, num_actions):
    if state not in Q:
        Q[state] = torch.zeros(num_actions, dtype=torch.float32)
    return Q[state]


def train_sarsa_with_history(
    num_episodes=10000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.9995,
    epsilon_min=0.01,
    seed=42,
    snapshot_interval=100
):
    random.seed(seed)
    torch.manual_seed(seed)

    env = ChainEnvWithHistory()
    Q = {}   # key: (position, history_tuple), value: tensor([Q(a=0), Q(a=1)])

    episode_rewards = []
    epsilon_history = []
    episode_log_rows = []
    q_snapshot_rows = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon, env.num_actions)

        total_reward = 0
        step_count = 0

        while True:
            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1

            q_state = get_q_row(Q, state, env.num_actions)

            if done:
                td_target = reward
                td_error = td_target - q_state[action]
                q_state[action] = q_state[action] + alpha * td_error
                break

            next_action = epsilon_greedy(Q, next_state, epsilon, env.num_actions)
            q_next = get_q_row(Q, next_state, env.num_actions)

            # SARSA update
            td_target = reward + gamma * q_next[next_action]
            td_error = td_target - q_state[action]
            q_state[action] = q_state[action] + alpha * td_error

            state = next_state
            action = next_action

        episode_rewards.append(total_reward)
        epsilon_history.append(epsilon)
        episode_log_rows.append([episode, total_reward, epsilon, step_count])

        if episode % snapshot_interval == 0 or episode == 1 or episode == num_episodes:
            for state_key in sorted(Q.keys(), key=lambda x: (x[0], len(x[1]), x[1])):
                qvals = Q[state_key]
                pos, hist = state_key
                state_name = f"x{pos+1}|hist={list(hist)}"
                q_snapshot_rows.append([
                    episode,
                    state_name,
                    float(qvals[0].item()),
                    float(qvals[1].item())
                ])

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, episode_rewards, epsilon_history, episode_log_rows, q_snapshot_rows


def evaluate_policy(Q):
    env = ChainEnvWithHistory()
    state = env.reset()
    actions_taken = []
    total_reward = 0
    visited_states = []

    while True:
        visited_states.append(state)

        if state not in Q:
            action = 0
        else:
            action = int(torch.argmax(Q[state]).item())

        actions_taken.append(action)

        next_state, reward, done = env.step(action)
        total_reward += reward

        if done:
            visited_states.append(next_state)
            break

        state = next_state

    return visited_states, actions_taken, total_reward


def save_episode_log(log_rows, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "epsilon", "steps"])
        writer.writerows(log_rows)


def save_q_snapshot(snapshot_rows, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "state", "Q_action_0", "Q_action_1"])
        writer.writerows(snapshot_rows)


def save_q_table(Q, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "Q_action_0", "Q_action_1", "best_action", "best_q"])

        for state_key in sorted(Q.keys(), key=lambda x: (x[0], len(x[1]), x[1])):
            qvals = Q[state_key]
            q0 = float(qvals[0].item())
            q1 = float(qvals[1].item())
            best_action = int(torch.argmax(qvals).item())
            best_q = float(torch.max(qvals).item())

            pos, hist = state_key
            state_name = f"x{pos+1}|hist={list(hist)}"

            writer.writerow([state_name, q0, q1, best_action, best_q])


def save_policy_text(visited_states, actions, total_reward, Q, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Greedy policy result ===\n")
        f.write(f"Visited states: {visited_states}\n")
        f.write(f"Actions taken: {actions}\n")
        f.write(f"Total reward: {total_reward}\n\n")

        f.write("=== Greedy action by augmented state ===\n")
        for state_key in sorted(Q.keys(), key=lambda x: (x[0], len(x[1]), x[1])):
            qvals = Q[state_key]
            best_action = int(torch.argmax(qvals).item())
            pos, hist = state_key
            f.write(
                f"x{pos+1}|hist={list(hist)}: "
                f"best action = {best_action}, Q = {qvals.tolist()}\n"
            )


def plot_reward_curve(rewards, save_path, ma_window=100):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, label="Episode reward")

    ma = moving_average(rewards, ma_window)
    if ma:
        plt.plot(
            range(ma_window, len(rewards) + 1),
            ma,
            label=f"Moving average ({ma_window})"
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SARSA Reward Curve (History Included)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_epsilon_curve(epsilon_history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epsilon_history) + 1), epsilon_history)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("..", "results", "sarsa_history", timestamp)
    ensure_dir(result_dir)

    Q, rewards, epsilon_history, episode_log_rows, q_snapshot_rows = train_sarsa_with_history(
        num_episodes=10000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        seed=42,
        snapshot_interval=100
    )

    visited_states, best_actions, best_reward = evaluate_policy(Q)
    conv_episode = estimate_convergence(rewards, window=100, tolerance=1e-3)

    print("=== Number of learned augmented states ===")
    print(len(Q))

    print("\n=== Greedy policy result ===")
    print("Visited states:", visited_states)
    print("Actions taken:", best_actions)
    print("Total reward:", best_reward)

    print("\n=== Some learned augmented states ===")
    for state_key in sorted(Q.keys(), key=lambda x: (x[0], len(x[1]), x[1]))[:20]:
        print(state_key, Q[state_key].tolist())

    print("\n=== Convergence estimate ===")
    print("Estimated convergence episode:", conv_episode)

    save_episode_log(
        episode_log_rows,
        os.path.join(result_dir, "episode_log.csv")
    )

    save_q_snapshot(
        q_snapshot_rows,
        os.path.join(result_dir, "q_snapshot.csv")
    )

    save_q_table(
        Q,
        os.path.join(result_dir, "q_table_final.csv")
    )

    save_policy_text(
        visited_states,
        best_actions,
        best_reward,
        Q,
        os.path.join(result_dir, "policy_log.txt")
    )

    with open(os.path.join(result_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Summary ===\n")
        f.write(f"Estimated convergence episode: {conv_episode}\n")
        f.write(f"Visited states: {visited_states}\n")
        f.write(f"Final greedy actions: {best_actions}\n")
        f.write(f"Final greedy reward: {best_reward}\n\n")
        f.write("[Interpretation]\n")
        f.write("This version includes action history in the state.\n")
        f.write("So x6 with history [0,1,0,1,0] and x6 with other histories are treated as different states.\n")

    plot_reward_curve(
        rewards,
        os.path.join(result_dir, "reward_curve.png"),
        ma_window=100
    )

    plot_epsilon_curve(
        epsilon_history,
        os.path.join(result_dir, "epsilon_curve.png")
    )

    print("\nSaved logs and plots to:", result_dir)