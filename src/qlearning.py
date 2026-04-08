import os
import csv
import random
from datetime import datetime

import torch
import matplotlib.pyplot as plt


class ChainEnv:
    """
    원래 문제의 reward 구조는 그대로 유지
    state만 (현재 위치, 이전 action history) 형태로 확장해서 사용
    """

    def __init__(self):
        self.num_states = 6
        self.num_actions = 2
        self.reset()

    def reset(self):
        self.state = 0   # x1 -> index 0
        self.prev_actions = []
        self.done = False
        return self._get_augmented_state()

    def _get_augmented_state(self):
        return (self.state, tuple(self.prev_actions))

    def step(self, action):
        if self.done:
            raise ValueError("Episode is done. Call reset().")

        current_state = self.state
        reward = 0

        # 마지막 상태 x6에서의 행동
        if current_state == 5:
            if action == 1:
                reward = 1
            else:
                if self.prev_actions == [0, 1, 0, 1, 0]:
                    reward = 1000
                else:
                    reward = 0

            self.done = True
            next_state = self._get_augmented_state()
            return next_state, reward, self.done

        # x1 ~ x5 에서의 행동
        if action == 1:
            reward = 1
        else:
            reward = 0

        self.prev_actions.append(action)
        self.state += 1
        next_state = self._get_augmented_state()

        return next_state, reward, self.done


class QTable:
    """
    history 포함 state를 dict로 관리
    key: (state_idx, history_tuple)
    value: torch.tensor([Q(s,0), Q(s,1)])
    """
    def __init__(self, num_actions=2):
        self.num_actions = num_actions
        self.table = {}

    def get(self, state):
        if state not in self.table:
            self.table[state] = torch.zeros(self.num_actions, dtype=torch.float32)
        return self.table[state]

    def best_action(self, state):
        return int(torch.argmax(self.get(state)).item())

    def max_value(self, state):
        return float(torch.max(self.get(state)).item())

    def items(self):
        return self.table.items()


def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    return Q.best_action(state)


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


def train_qlearning_history(
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

    env = ChainEnv()
    Q = QTable(num_actions=2)

    episode_rewards = []
    epsilon_history = []
    episode_log_rows = []
    q_snapshot_rows = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        step_count = 0

        while True:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = env.step(action)

            total_reward += reward
            step_count += 1

            q_values = Q.get(state)

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * Q.max_value(next_state)

            td_error = td_target - q_values[action]
            q_values[action] = q_values[action] + alpha * td_error

            if done:
                break

            state = next_state

        episode_rewards.append(total_reward)
        epsilon_history.append(epsilon)
        episode_log_rows.append([episode, total_reward, epsilon, step_count])

        if episode % snapshot_interval == 0 or episode == 1 or episode == num_episodes:
            for state_key, q_tensor in Q.items():
                state_idx, history = state_key
                q_snapshot_rows.append([
                    episode,
                    f"x{state_idx+1}|{list(history)}",
                    float(q_tensor[0].item()),
                    float(q_tensor[1].item())
                ])

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, episode_rewards, epsilon_history, episode_log_rows, q_snapshot_rows


def evaluate_policy(Q):
    env = ChainEnv()
    state = env.reset()
    actions_taken = []
    total_reward = 0

    while True:
        action = Q.best_action(state)
        actions_taken.append(action)

        next_state, reward, done = env.step(action)
        total_reward += reward

        if done:
            break

        state = next_state

    return actions_taken, total_reward


def save_episode_log(log_rows, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "epsilon", "steps"])
        writer.writerows(log_rows)


def save_q_snapshot(snapshot_rows, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "state_with_history", "Q_action_0", "Q_action_1"])
        writer.writerows(snapshot_rows)


def save_q_table(Q, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["state_with_history", "Q_action_0", "Q_action_1", "best_action", "best_q"])

        for state_key, q_tensor in sorted(Q.items(), key=lambda x: (x[0][0], len(x[0][1]), x[0][1])):
            state_idx, history = state_key
            q0 = float(q_tensor[0].item())
            q1 = float(q_tensor[1].item())
            best_action = int(torch.argmax(q_tensor).item())
            best_q = float(torch.max(q_tensor).item())
            writer.writerow([f"x{state_idx+1}|{list(history)}", q0, q1, best_action, best_q])


def save_policy_text(actions, total_reward, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Greedy policy result ===\n")
        f.write(f"Actions taken: {actions}\n")
        f.write(f"Total reward: {total_reward}\n")


def plot_reward_curve(rewards, save_path, ma_window=100):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, label="Episode reward")

    ma = moving_average(rewards, ma_window)
    if ma:
        plt.plot(range(ma_window, len(rewards) + 1), ma, label=f"Moving average ({ma_window})")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Reward Curve (History State)")
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


def plot_value_function_mean(q_snapshot_rows, save_path):
    by_episode = {}
    for row in q_snapshot_rows:
        episode, state_name, q0, q1 = row
        if episode not in by_episode:
            by_episode[episode] = []
        by_episode[episode].append(max(q0, q1))

    episodes = sorted(by_episode.keys())
    vf_mean = [sum(by_episode[e]) / len(by_episode[e]) for e in episodes]

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, vf_mean)
    plt.xlabel("Episode")
    plt.ylabel("Mean max Q(s,a)")
    plt.title("Value Function Mean (History State)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("..", "results", "qlearning_history", timestamp)
    ensure_dir(result_dir)

    Q, rewards, epsilon_history, episode_log_rows, q_snapshot_rows = train_qlearning_history(
        num_episodes=10000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        seed=42,
        snapshot_interval=100
    )

    best_actions, best_reward = evaluate_policy(Q)
    conv_episode = estimate_convergence(rewards, window=100, tolerance=1e-3)

    print("=== Greedy policy result ===")
    print("Actions taken:", best_actions)
    print("Total reward:", best_reward)
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
        best_actions,
        best_reward,
        os.path.join(result_dir, "policy_log.txt")
    )

    with open(os.path.join(result_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Summary ===\n")
        f.write(f"Estimated convergence episode: {conv_episode}\n")
        f.write(f"Final greedy actions: {best_actions}\n")
        f.write(f"Final greedy reward: {best_reward}\n")
        f.write("\n[Interpretation]\n")
        f.write("History was added to the state representation for Q-learning.\n")

    plot_reward_curve(
        rewards,
        os.path.join(result_dir, "reward_curve.png"),
        ma_window=100
    )

    plot_epsilon_curve(
        epsilon_history,
        os.path.join(result_dir, "epsilon_curve.png")
    )

    plot_value_function_mean(
        q_snapshot_rows,
        os.path.join(result_dir, "value_function_mean.png")
    )

    print("\nSaved logs and plots to:", result_dir)