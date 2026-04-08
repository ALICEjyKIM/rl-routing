import os
import csv
import random
from datetime import datetime

import torch
import matplotlib.pyplot as plt


class ChainEnv:
    """
    source 기반 환경
    states: x1 -> x2 -> x3 -> x4 -> x5 -> x6
    actions: 0, 1
    transition: deterministic (다음 상태로 한 칸 이동)

    reward:
        if action == 1:
            reward = 1
        elif action == 0:
            if current_state == x6 and prev_actions == [0, 1, 0, 1, 0]:
                reward = 1000
            else:
                reward = 0

    주의:
    마지막 상태 x6에서 action을 한 번 더 선택하는 구조로 해석하면,
    그 전에 누적된 5개의 action 패턴을 보고 최종 보상을 주는 형태가 된다.
    """

    def __init__(self):
        self.num_states = 6
        self.num_actions = 2
        self.reset()

    def reset(self):
        self.state = 0   # x1 -> index 0
        self.prev_actions = []
        self.done = False
        return self.state

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
            next_state = current_state
            return next_state, reward, self.done

        # x1 ~ x5 에서의 행동
        if action == 1:
            reward = 1
        else:
            reward = 0

        self.prev_actions.append(action)
        self.state += 1
        next_state = self.state

        return next_state, reward, self.done


def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, Q.shape[1] - 1)
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
    """
    최근 window 구간 평균 reward의 변화가 tolerance보다 작아지는
    첫 시점을 대략적인 수렴 시점으로 본다.
    """
    if len(episode_rewards) < 2 * window:
        return None

    ma = moving_average(episode_rewards, window)
    for i in range(1, len(ma)):
        if abs(ma[i] - ma[i - 1]) < tolerance:
            return i + window
    return None


def train_sarsa(
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

    # Q-table: [num_states, num_actions]
    Q = torch.zeros((env.num_states, env.num_actions), dtype=torch.float32)

    episode_rewards = []
    epsilon_history = []
    episode_log_rows = []
    q_snapshot_rows = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        total_reward = 0
        step_count = 0

        while True:
            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1

            if done:
                td_target = reward
                td_error = td_target - Q[state, action]
                Q[state, action] = Q[state, action] + alpha * td_error
                break

            next_action = epsilon_greedy(Q, next_state, epsilon)

            # SARSA update
            td_target = reward + gamma * Q[next_state, next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error

            state = next_state
            action = next_action

        episode_rewards.append(total_reward)
        epsilon_history.append(epsilon)
        episode_log_rows.append([episode, total_reward, epsilon, step_count])

        if episode % snapshot_interval == 0 or episode == 1 or episode == num_episodes:
            for s in range(Q.shape[0]):
                q_snapshot_rows.append([
                    episode,
                    f"x{s+1}",
                    float(Q[s, 0].item()),
                    float(Q[s, 1].item())
                ])

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, episode_rewards, epsilon_history, episode_log_rows, q_snapshot_rows


def evaluate_policy(Q):
    env = ChainEnv()
    state = env.reset()
    actions_taken = []
    total_reward = 0

    while True:
        action = int(torch.argmax(Q[state]).item())
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
        writer.writerow(["episode", "state", "Q_action_0", "Q_action_1"])
        writer.writerows(snapshot_rows)


def save_q_table(Q, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "Q_action_0", "Q_action_1", "best_action", "best_q"])
        for s in range(Q.shape[0]):
            q0 = float(Q[s, 0].item())
            q1 = float(Q[s, 1].item())
            best_action = int(torch.argmax(Q[s]).item())
            best_q = float(torch.max(Q[s]).item())
            writer.writerow([f"x{s+1}", q0, q1, best_action, best_q])


def save_policy_text(actions, total_reward, Q, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Greedy policy result ===\n")
        f.write(f"Actions taken: {actions}\n")
        f.write(f"Total reward: {total_reward}\n\n")

        f.write("=== Greedy action by state ===\n")
        for s in range(Q.shape[0]):
            best_action = int(torch.argmax(Q[s]).item())
            q_values = Q[s].tolist()
            f.write(f"x{s+1}: best action = {best_action}, Q = {q_values}\n")


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
    plt.title("SARSA Reward Curve")
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


def plot_q_value_by_state(q_snapshot_rows, save_path):
    """
    state별로 episode에 따라 Q(s,0), Q(s,1)이 어떻게 변하는지 한 그래프에 저장
    """
    state_dict = {}
    for row in q_snapshot_rows:
        episode, state, q0, q1 = row
        if state not in state_dict:
            state_dict[state] = {"episode": [], "q0": [], "q1": []}
        state_dict[state]["episode"].append(episode)
        state_dict[state]["q0"].append(q0)
        state_dict[state]["q1"].append(q1)

    plt.figure(figsize=(10, 6))
    for state in sorted(state_dict.keys()):
        plt.plot(state_dict[state]["episode"], state_dict[state]["q0"], label=f"{state}-a0")
        plt.plot(state_dict[state]["episode"], state_dict[state]["q1"], linestyle="--", label=f"{state}-a1")

    plt.xlabel("Episode")
    plt.ylabel("Q value")
    plt.title("Q-value Change by State")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("..", "results", "sarsa", timestamp)
    ensure_dir(result_dir)

    Q, rewards, epsilon_history, episode_log_rows, q_snapshot_rows = train_sarsa(
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

    print("=== Learned Q-table ===")
    print(Q)

    print("\n=== Greedy policy result ===")
    print("Actions taken:", best_actions)
    print("Total reward:", best_reward)

    print("\n=== Greedy action by state ===")
    for s in range(Q.shape[0]):
        print(f"x{s+1}: best action = {int(torch.argmax(Q[s]).item())}, Q = {Q[s].tolist()}")

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
        best_actions,
        best_reward,
        Q,
        os.path.join(result_dir, "policy_log.txt")
    )

    with open(os.path.join(result_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Summary ===\n")
        f.write(f"Estimated convergence episode: {conv_episode}\n")
        f.write(f"Final greedy actions: {best_actions}\n")
        f.write(f"Final greedy reward: {best_reward}\n")
        f.write("\n[Interpretation]\n")
        f.write("If greedy reward is far below 1002, the learned policy did not reach the desired pattern.\n")

    plot_reward_curve(
        rewards,
        os.path.join(result_dir, "reward_curve.png"),
        ma_window=100
    )

    plot_epsilon_curve(
        epsilon_history,
        os.path.join(result_dir, "epsilon_curve.png")
    )

    plot_q_value_by_state(
        q_snapshot_rows,
        os.path.join(result_dir, "q_value_by_state.png")
    )

    print("\nSaved logs and plots to:", result_dir)