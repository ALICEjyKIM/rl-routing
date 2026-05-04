import csv
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

import torch.serialization

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.serialization.add_safe_globals([np._core.multiarray.scalar])

# 하이퍼파라미터
gamma = 0.99
lr = 0.0003
patience = 50  # 성능 개선이 없을 때 기다릴 에피소드 수
target_reward = 200  # LunarLander 해결 기준 (보통 200점 이상)

log_path = os.path.join(os.path.dirname(__file__), "A2C_training_log.csv")

# nn.Module을 상속한 class ActorCritic
# dim 하나당 state/action 하나
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        # 상속받은거 쓸수있게 setting
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        # dimension 당 하나의 평균 출력
        self.mu = nn.Linear(128, action_dim)
        # actor쪽: 행동분포의 표준편차를 만들기 위한 값
        # std 양수여야 해서 직접 std 학습X. 일단 log_std 학습후 forward에서 변경
        # 0으로 채워진 텐서 만들기 (1행 action_dim열)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        # critic 쪽: 중간feature 128개 보고 현재 상태 가치값을 숫자 하나로 출력
        self.value = nn.Linear(128, 1)

    # x는 입력 state
    def forward(self, x):
        # 입력 state를 128개 중간 특징으로 바꿔라
        x = self.fc(x)
        # actor: 128개 특징 받아 action_dim개 숫자 출력 후, 그값을 -1~1로 제한
        # 즉, 각 action 성분의 평균값을 만든다는 뜻
        mu = torch.tanh(self.mu(x))
        # actor 표준편차 양수화: 얼마나 랜덤하게 뽑을 것인가? 
        std = torch.exp(self.log_std)
        # critic: 128개 특징을 보고 현재 상태의 가치 V(s)를 숫자 하나로 예측
        value = self.value(x)
        return mu, std, value

# 학습 중간 상태를 파일로 저장하는 함수
def save_model(model, optimizer, episode, best_reward, path):
    """
    모델 가중치뿐만 아니라 학습 상태 전체를 저장
    """
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_reward': best_reward
    }
    torch.save(checkpoint, path)
    print(f"--- Checkpoint saved at episode {episode} (Best: {best_reward:.2f}) ---")

def train():
    # 강화학습 환경을 만드는 코드
    env = gym.make("LunarLanderContinuous-v3")
    # ActorCritic 모델을 생성하는 부분: state, action 차원
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    # 모델의 가중치를 실제로 업데이트하는 도구
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 로그 파일 생성
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["episode", "avg_reward", "best_reward", "actor_loss", "value_loss"])

    all_rewards, avg_rewards = [], []
    actor_losses, value_losses, avg_actor_losses, avg_value_losses = [], [], [], []
    # 지금까지의 최고 reward를 초기화하는 부분: 처음에는 최고 보상이 아직 없기 때문
    best_reward = -float('inf')
    # early stopping에 쓰임
    no_improve_count = 0

    for episode in range(10000):
        # 환경을 초기화하는 코드: 원래 env.reset()는 두개 반환하는데 info는 안써서 _
        state, _ = env.reset()
        # log_probs = 선택한 action의 로그확률
        # values    = Critic이 예측한 V(s)
        # rewards   = 실제 받은 reward
        # masks     = 에피소드가 끝났는지 여부
        log_probs, values, rewards, masks = [], [], [], []
        entropy = 0

        # 한 에피 당 1000 step까지 행동하겠다
        for t in range(1000):
            # 현재 state를 PyTorch 텐서로 바꾸는 코드
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # 현재 상태를 ActorCritic 모델에 넣는 부분
            mu, std, value = model(state_tensor)
            
            # 정규분포 만드는 코드
            dist = Normal(mu, std)
            # 정규분포에서 실제 action을 샘플링하는 부분
            action = dist.sample()
            
            # 환경에 action을 넣고 한 step 진행하는 코드
            # action.numpy()[0]은 PyTorch tensor인 action을 Gym 환경이 받을 수 있는 NumPy 형태로 바꾸는 것
            # unsqueeze(0)로 추가했던 batch 차원을 다시 제거
            # action을 환경에 넣으면 다음 정보를 줌
            # next_state = action 이후의 다음 상태
            # reward = 이번 행동으로 받은 보상
            # terminated/truncated = 에피소드 종료 여부
            next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
            # 둘중 하나면 종료: terminated = 착륙 성공실패처럼 게임 규칙상 종료 / truncated  = 최대 step 도달처럼 시간 제한 종료
            done = terminated or truncated

            # 이번에 선택한 action의 로그확률을 저장하는 코드. Actor loss 계산시 필요
            # 왜 sum(dim=-1)을 하냐면 action이 여러 차원일 수 있기 때문
            log_probs.append(dist.log_prob(action).sum(dim=-1))
            # Critic이 예측한 현재 상태 가치 V(s)를 저장
            values.append(value)
            # 환경에서 받은 실제 reward를 저장. 나중에 loss 계산할 때 tensor끼리 계산하기 편하게 하려고 Tensor
            rewards.append(torch.FloatTensor([reward]))
            # 다음 상태가 이어지는지 끝났는지를 저장하는 값
            masks.append(torch.FloatTensor([1 - done]))

            # 현재 상태를 다음 상태로 바꾸는 코드
            state = next_state
            # 착륙 성공실패했거나 시간 제한으로 끝났으면 더 이상 action을 뽑지 않고 다음 episode로 넘어가기
            if done: break

        # Returns & Advantage 계산
            # reward들을 뒤에서부터 누적해서 returns 계산
            # returns와 values 차이로 advantage 계산
            # actor_loss, critic_loss 계산
            # gradient 계산
            # optimizer가 모델 파라미터 업데이트
        # 마지막 next_state의 value를 계산하는 코드
        # next_state를 PyTorch tensor로 바꾸고 batch 차원을 추가 (state_dim,) > (1, state_dim)
        # model(..)[2]는 value
        next_v = model(torch.FloatTensor(next_state).unsqueeze(0))[2]
        # 각 step마다의 누적 보상값을 여기에 저장
        returns = []

        # R을 마지막 다음 상태의 value로 시작
        # detach()는 gradient 계산에서 떼어낸다는 뜻 = R은 target 계산용 숫자로만 쓰겠다
        # next_v로 시작하는 이유는 에피소드 안끝났을 수도 있어서 
        # 마지막 이후 미래가치를 critic이 예측한 V(next_state)로 이어붙여서 계산
        R = next_v.detach()
        # rewards와 masks를 뒤에서부터 하나씩 꺼내는 반복문
        for r, m in zip(reversed(rewards), reversed(masks)):
            # 현재 return = 현재 reward + 할인율 × 다음 return × mask
            # 에피소드가 끝났으면 다음 return을 더하면 안 되니까 m을 곱함
            R = r + gamma * R * m
            # 뒤에서부터 계산하고 있기 때문에 방금 계산한 R을 리스트 맨앞에 넣기
            returns.insert(0, R)
            
        # 리스트에 들어 있던 return 텐서들을 하나의 텐서로 합치기
        returns = torch.stack(returns).squeeze()
        # 각 step에서 Critic이 예측했던 value들을 하나의 텐서로 합치기
        values = torch.stack(values).squeeze()
        # 각 step에서 선택한 action의 log probability도 하나의 텐서로 합치기
        log_probs = torch.stack(log_probs).squeeze()
        # Advantage 계산
        advantage = returns - values

        # optimizer는 기본적으로 loss를 최소화하니까 앞에 -
        # advantage.detach()는 Actor를 업데이트할 때 advantage 값 자체는 고정된 점수처럼 쓰겠다는 뜻
        # Actor loss를 계산할 때 Critic 쪽 values까지 같이 건드리지 않게 막는 것
        actor_loss = -(log_probs * advantage.detach()).mean()
        # Critic loss를 계산: V(s)를 맞히는 역할
        critic_loss = F.mse_loss(values, returns)
        # 전체 loss
        loss = actor_loss + critic_loss

        # 이전에 계산된 gradient를 초기화: PyTorch는 gradient 자동 누적하므로 매번 업데이트 전에 이전 gradient 지워야함
        optimizer.zero_grad()
        # 전체 loss에 대해 gradient를 계산: 모델 안의 파라미터들이 loss를 줄이려면 어느 방향으로 바뀌어야 하는지 계산
        loss.backward()
        # optimizer가 실제로 모델 파라미터를 업데이트
        optimizer.step()

        # 결과 기록
        episode_reward = sum(rewards).item()
        all_rewards.append(episode_reward)
        avg_reward = np.mean(all_rewards[-10:])
        avg_rewards.append(avg_reward)
        
        actor_losses.append(actor_loss.item())
        avg_actor_loss = np.mean(actor_losses[-10:])
        avg_actor_losses.append(avg_actor_loss)

        value_losses.append(critic_loss.item())
        avg_value_loss = np.mean(value_losses[-10:])
        avg_value_losses.append(avg_value_loss)

        if episode % 100 == 0:
            print(f"Ep {episode} | Avg Rew: {avg_reward:.1f} | BEST Rew: {best_reward:.1f} | A_Loss: {actor_loss.item():.3f} | V_Loss: {critic_loss.item():.3f}")
            with open(log_path, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    episode,
                    f"{avg_reward:.1f}",
                    f"{best_reward:.1f}",
                    f"{actor_loss.item():.3f}",
                    f"{critic_loss.item():.3f}",
                ])

        # --- Early Stopping Logic ---
        if avg_reward > best_reward:
            best_reward = avg_reward
            no_improve_count = 0
            save_model(model, optimizer, episode, best_reward, "A2C_basic_best.pth")
        else:
            no_improve_count += 1

        if no_improve_count >= patience and avg_reward >= target_reward:
            print(f"Early Stopping! Target reached at episode {episode}")
            break

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Reward
    axs[0].plot(all_rewards, alpha=0.3, color='blue')
    axs[0].plot(avg_rewards, color='red', label='Moving Avg (10)')
    axs[0].axhline(y=target_reward, color='green', linestyle='--')
    axs[0].set_title('Episode Reward')
    axs[0].legend()

    # 2. Actor Loss
    axs[1].plot(actor_losses, color='orange')
    axs[1].plot(avg_actor_losses, color='red', label='Moving Avg (10)')
    axs[1].set_title('Actor Loss (Policy Gradient)')

    # 3. Value Loss
    axs[2].plot(value_losses, color='purple')
    axs[2].plot(avg_value_losses, color='red', label='Moving Avg (10)')
    axs[2].set_title('Value Loss (MSE)')

    plt.tight_layout()
    plt.savefig('A2C_basic_metrics.png')
    print("Metrics plot saved as 'A2C_basic_metrics.png'")
    plt.show()

def inference(model_path="A2C_basic_best.pth", episodes=5):
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")

    model = ActorCritic(
        env.observation_space.shape[0],
        env.action_space.shape[0]
    )

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(1000):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                mu, std, value = model(state_tensor)

            action = mu

            next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
            done = terminated or truncated

            episode_reward += reward
            state = next_state

            if done:
                break

        print(f"Inference Episode {episode + 1} | Reward: {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    train()
    inference()
