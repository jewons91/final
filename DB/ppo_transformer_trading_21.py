# state에 평단가 추가 및 보상 수치 조정 포트폴리오 리턴이랑 invalid action
# gamma 0.8 => 0.7 => 0.6
# 하락장
# 변동률 제거
# 데이터 변경
# State에 Unscaled Close
# 행동제약
# Reward 수정
# 전체 Standard/ gamma 0.5 / epoch 30 / epsilon 0.5 / 행동제약 품 / episode 3000 / reward 변경
# Sell 할때 portfolio value 상승시 더 큰 보상
# portfolio value -0.01 일때 buy, hold면 더 큰 손실

import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch.nn.functional as F
import warnings
import math

# UserWarning 무시 (필요 시 제거 가능)
warnings.filterwarnings("ignore", category=UserWarning)

# 디바이스 설정 (GPU 사용 여부 확인)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 포지셔널 인코딩 계산
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 주파수의 스케일을 조정하기 위한 div_term 계산
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        # 포지셔널 인코딩 적용
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
        pe = pe.unsqueeze(0)  # 배치 차원 추가
        self.register_buffer('pe', pe)  # 학습되지 않는 버퍼로 등록

    def forward(self, x):
        """
        x: [batch_size, seq_length, embed_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
# Transformer 모듈 정의 (8개의 TransformerEncoderLayer 스택)
class Transformer(nn.Module):
    def __init__(self, input_dim, seq_length, embed_dim=128, num_heads=4, num_layers=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.input_linear = nn.Linear(input_dim, embed_dim)
        
        # 포지셔널 인코딩 모듈 추가
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout, max_len=seq_length)
        
        # TransformerEncoderLayer에 batch_first=True 설정
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True  # batch_first 설정
        )
        
        # 8개의 TransformerEncoderLayer를 쌓기 위해 num_layers=8로 설정
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(embed_dim)  # LayerNorm 사용

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_dim]
        """
        x = self.input_linear(x)  # [batch_size, seq_length, embed_dim]
        x = self.pos_encoder(x)   # 포지셔널 인코딩 추가
        x = self.transformer_encoder(x)  # [batch_size, seq_length, embed_dim]
        x = self.output_linear(x)  # [batch_size, seq_length, embed_dim]
        x = self.activation(x)
        x = self.layer_norm(x[:, -1, :])  # LayerNorm 적용

        return x

# 다중 종목 주식 트레이딩 환경 정의
class MultiStockTradingEnv(gym.Env):
    def __init__(self, dfs, stock_dim=3,input_dim=7, initial_balance=5000000, max_stock=500, seq_length=20):
        super(MultiStockTradingEnv, self).__init__()
        self.dfs = dfs  # 종목별 데이터프레임 딕셔너리
        self.real_dfs = dfs.copy()
        self.stock_dim = stock_dim  # 종목 수
        self.initial_balance = initial_balance  # 초기 자본금 증가
        self.max_stock = max_stock  # 각 종목당 최대 보유 주식 수 증가
        self.transaction_fee = 0.00015  # 거래 수수료 0.25%
        self.national_tax = 0.0018  # 매도세 0.18%
        self.slippage = 0.0005  # 슬리피지 0.3%
        self.max_loss = 0.20  # 최대 허용 손실 (20%)
        self.seq_length = seq_length  # 시퀀스 길이
        self.input_dim = input_dim
        # 상태 공간: 시퀀스 길이에 따른 각 종목의 [Close, MA10, MA20, RSI, Volume, Bollinger Bands]와 보유 주식 수
        # 각 종목: (7 * seq_length) + 1 (보유 주식 수)
        # 총: stock_dim * (7 * seq_length +1) + 1 (balance)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.stock_dim * (self.input_dim * self.seq_length + 2) + 1,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([12] * self.stock_dim)  # 각 종목별로 11개의 액션

        # 초기화
        self.reset()

    def reset(self, data_range=None):
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.current_step = 0
        self.stock_owned = {ticker: {'quantity': 0, 'avg_price': 0} for ticker in self.dfs.keys()}
        self.stock_price = {}
        self.total_asset = []

        # 히스토리 초기화
        self.balance_history = [self.balance]
        self.portfolio_value_history = [self.portfolio_value]
        self.action_history = []
        self.price_history = {ticker: [] for ticker in self.dfs.keys()}
        self.trade_history = []  # 매수/매도 내역 저장

        # 각 종목의 데이터 슬라이스 적용
        if data_range is not None:
            start_idx, end_idx = data_range
            self.dfs = {ticker: df.iloc[start_idx:end_idx].reset_index(drop=True) for ticker, df in self.real_dfs.items()}
        # else:
        #     # 전체 데이터 사용
        #     self.dfs = {ticker: df.reset_index(drop=True) for ticker, df in self.dfs.items()} 
        
        
        # 각 종목의 최대 스텝 수 계산
        self.max_steps = min(len(df) for df in self.dfs.values()) - self.seq_length - 1
        if self.max_steps <= 0:
            raise ValueError("데이터프레임의 길이가 시퀀스 길이보다 짧습니다.")

        # 각 종목의 현재 인덱스 초기화
        self.data_indices = {ticker: self.seq_length for ticker in self.dfs.keys()}  # 시퀀스 길이만큼 초기 인덱스

        return self._next_observation()

    def _next_observation(self):
        obs = []
        for ticker, df in self.dfs.items():
            idx = self.data_indices[ticker]
            if idx < self.seq_length:
                # 충분한 시퀀스가 없으면 제로 패딩
                seq = df.loc[:idx, ['Close', 'Low', 'MA10', 'MA20', 'RSI', 'Volume', 'Upper_Band', 'Lower_Band']].values
                pad_length = self.seq_length - seq.shape[0]
                if pad_length > 0:
                    pad = np.zeros((pad_length, self.input_dim))
                    seq = np.vstack((pad, seq))
            else:
                # 시퀀스 슬라이싱
                seq = df.loc[idx - self.seq_length:idx - 1, ['Close', 'Low', 'MA10', 'MA20', 'RSI', 'Volume', 'Upper_Band', 'Lower_Band']].values
            obs.extend(seq.flatten())  # [seq_length * 7]
            obs.append(self.stock_owned[ticker]['quantity'])  # [1]
            obs.append(self.stock_owned[ticker]['avg_price'])

            # 현재 가격 저장 (iloc 사용)
            self.stock_price[ticker] = df.iloc[idx]['Close_unscaled']

        # 잔고 추가
        obs.append(self.balance)  # [1]

        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        done = False
        total_reward = 0
        invalid_action_penalty = -1000  # 불가능한 행동에 대한 페널티
        self.action_history.append(actions)
        trade_info = []  # 현재 스텝의 거래 내역

        prev_portfolio_value = self.portfolio_value  # 이전 포트폴리오 가치 저장

        # 액션 맵핑 정의
        action_mapping = {
            0: ('sell', 1.0),
            1: ('sell', 0.75),
            2: ('sell', 0.5),
            3: ('sell', 0.25),
            4: ('hold', 0.0),
            5: ('buy', 0.25),
            6: ('buy', 0.5),
            7: ('buy', 0.75),
            8: ('buy', 1.0)
        }

        # 종목별로 행동 수행
        for i, (ticker, df) in enumerate(self.dfs.items()):
            action = actions[i]
            idx = self.data_indices[ticker]

            # 인덱스가 데이터프레임의 범위를 벗어나면 에피소드를 종료
            if idx >= len(df):
                done = True
                trade_info.append(f"Ticker {ticker} reached end of data. Ending episode.")
                break

            actual_price = df.iloc[idx]['Close_unscaled']

            # 액션 맵핑을 통해 행동 타입과 비율 얻기
            action_type, proportion = action_mapping.get(action, ('hold', 0.0))

            # 슬리피지 적용
            if action_type == 'buy':
                adjusted_price = actual_price * (1 + self.slippage)
            elif action_type == 'sell':
                adjusted_price = actual_price * (1 - self.slippage)
            else:
                adjusted_price = actual_price

            # 거래 수수료 계산
            buy_fee = adjusted_price * self.transaction_fee
            sell_fee = adjusted_price * (self.transaction_fee + self.national_tax)

            # buy_amount과 sell_amount 초기화
            buy_amount = 0
            sell_amount = 0

            # 행동에 따른 포트폴리오 업데이트 및 보상 계산
            reward = 0  # 각 종목별 보상 초기화
            if action_type == 'sell':
                if self.stock_owned[ticker]['quantity'] > 0:
                    sell_amount = int(self.stock_owned[ticker]['quantity'] * proportion)
                    sell_amount = max(1, sell_amount)  # 최소 1주 매도
                    sell_amount = min(sell_amount, self.stock_owned[ticker]['quantity'])  # 보유 주식 수 초과 방지
                    proceeds = adjusted_price * sell_amount - sell_fee * sell_amount
                    self.balance += proceeds
                    # 이익 또는 손실 계산
                    profit = (adjusted_price - self.stock_owned[ticker]['avg_price']) * sell_amount - sell_fee * sell_amount
                    reward = profit  # 매도 시 보상은 이익 또는 손실
                    self.stock_owned[ticker]['quantity'] -= sell_amount
                    if self.stock_owned[ticker]['quantity'] == 0:
                        self.stock_owned[ticker]['avg_price'] = 0
                    trade_info.append(f"Sell {sell_amount} of {ticker} at {adjusted_price:.2f}")
                else:
                    # 보유한 주식이 없으면 페널티 부여
                    reward = invalid_action_penalty
                    trade_info.append(f"Cannot Sell {ticker} (No holdings)")
            elif action_type == 'buy':
                max_can_buy = min(
                    self.max_stock - self.stock_owned[ticker]['quantity'],
                    int(self.balance // (adjusted_price + buy_fee))
                )
                buy_amount = int(max_can_buy * proportion)
                buy_amount = max(1, buy_amount)  # 최소 1주 매수
                buy_amount = min(buy_amount, self.max_stock - self.stock_owned[ticker]['quantity'], 
                                 int(self.balance // (adjusted_price + buy_fee)))
                if buy_amount > 0:
                    cost = adjusted_price * buy_amount + buy_fee * buy_amount
                    self.balance -= cost
                    # 평균 매수가격 업데이트
                    total_quantity = self.stock_owned[ticker]['quantity'] + buy_amount
                    if total_quantity > 0:
                        self.stock_owned[ticker]['avg_price'] = (
                            (self.stock_owned[ticker]['avg_price'] * self.stock_owned[ticker]['quantity'] + adjusted_price * buy_amount)
                            / total_quantity
                        )
                    self.stock_owned[ticker]['quantity'] = total_quantity
                    reward = 0  # 매수 시 보상 없음
                    trade_info.append(f"Buy {buy_amount} of {ticker} at {adjusted_price:.2f}")
                else:
                    reward = invalid_action_penalty  # 매수 불가 시 페널티
                    trade_info.append(f"Cannot Buy {ticker} (Insufficient balance or max stock)")
            else:
                trade_info.append(f"Hold {ticker}")

            # 보상 누적
            total_reward += reward

            # 가격 히스토리 업데이트
            self.price_history[ticker].append(actual_price)

            # 다음 인덱스로 이동
            self.data_indices[ticker] += 1

            # 디버깅을 위한 로그 출력
            print(f"Ticker: {ticker}, Action: {action_type}, Proportion: {proportion}, "
                  f"Buy/Sell Amount: {buy_amount if action_type == 'buy' else sell_amount if action_type == 'sell' else 0}, "
                  f"Balance: {self.balance:.2f}, Holdings: {self.stock_owned[ticker]['quantity']}")

            # 잔고와 보유 주식 수량이 음수가 되지 않도록 방지
            if self.balance < 0:
                print(f"Warning: Balance for {ticker} is negative! Setting to 0.")
                self.balance = 0
            if self.stock_owned[ticker]['quantity'] < 0:
                print(f"Warning: Holdings for {ticker} are negative! Setting to 0.")
                self.stock_owned[ticker]['quantity'] = 0
                self.stock_owned[ticker]['avg_price'] = 0

        # 현재 스텝의 거래 내역 저장 및 출력
        if trade_info:
            print(f"Episode {self.current_step} Step {self.current_step}:")
            for info in trade_info:
                print(info)
            print()

        self.trade_history.append(trade_info)

        # 현재 포트폴리오 가치 계산
        self.portfolio_value = self.balance + sum(
            self.stock_owned[ticker]['quantity'] * self.stock_price[ticker] for ticker in self.dfs.keys()
        )
        self.total_asset.append(self.portfolio_value)

        # 보상은 포트폴리오 가치의 비율적 변화량을 사용
        if prev_portfolio_value > 0:
            portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value  # 비율적 변화
        else:
            portfolio_return = 0  # 이전 포트폴리오 가치가 0일 경우

        # 보상 함수 개선: 수익률에 따라 보상을 증가
        for i, (ticker, df) in enumerate(self.dfs.items()):
            if portfolio_return > 0.01:
                if action_type == 'sell':
                    scaled_reward = portfolio_return * 30000
                else:
                    scaled_reward = portfolio_return * 15000
            elif portfolio_return < -0.01:
                if action_type in ('buy', 'hold'):
                    scaled_reward = portfolio_return * 30000
                else:
                    scaled_reward = portfolio_return * 15000
            else:
                scaled_reward = portfolio_return * 10000

            total_reward += scaled_reward  # 전체 보상에 추가

        # 포트폴리오 가치가 음수가 되지 않도록 보장
        if self.portfolio_value < 0:
            print(f"Error: Portfolio value is negative! Setting to 0.")
            self.portfolio_value = 0
            self.balance = 0  # 잔고도 0으로 설정
            done = True  # 에피소드를 종료

        # 최대 허용 손실 초과 시 종료
        if self.portfolio_value < self.initial_balance * (1 - self.max_loss):
            done = True

        # 최대 스텝 수 초과 시 종료
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # 히스토리 업데이트
        self.balance_history.append(self.balance)
        self.portfolio_value_history.append(self.portfolio_value)

        obs = self._next_observation()
        return obs, total_reward, done, {}

# PPO를 위한 액터-크리틱 신경망 정의
class ActorCritic(nn.Module):
    def __init__(self,  action_dim_list, seq_length=20, input_dim=7):
        super(ActorCritic, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.transformer = Transformer(input_dim=self.input_dim, seq_length=seq_length).to(device)  # input_dim=7 (Close, MA10, MA20, RSI, Volume, Bollinger Bands)
        self.policy_head = nn.ModuleList([nn.Linear(self.transformer.embed_dim, action_dim) for action_dim in action_dim_list])
        self.value_head = nn.Linear(self.transformer.embed_dim * len(action_dim_list), 1)  # embed_dim * stock_dim
        self.apply(self._weights_init)  # 가중치 초기화

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            # 정책 헤드의 바이어스를 0으로 초기화하여 행동 확률 균등화
            nn.init.zeros_(m.bias)
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        x: [batch_size, stock_dim * (7 * seq_length +1) +1]
        """
        stock_embeds = []
        for i in range(len(tickers)):
            # 각 종목의 시퀀스 데이터: [batch_size, 7 * seq_length]
            start = i * (self.input_dim * self.seq_length + 1)
            end = start + self.input_dim * self.seq_length
            seq = x[:, start:end]
            # Reshape to [batch_size, seq_length, 7]
            seq = seq.view(-1, self.seq_length, self.input_dim)
            embed = self.transformer(seq)  # [batch_size, embed_dim]
            stock_embeds.append(embed)
        # 정책 헤드는 각 embed를 처리하여 policy logits을 생성
        policy_logits = [head(embed) for embed, head in zip(stock_embeds, self.policy_head)]  # [batch_size, 9] * stock_dim
        # 가치 함수는 모든 embed를 합쳐 처리
        combined_embeds = torch.cat(stock_embeds, dim=1)  # [batch_size, embed_dim * stock_dim]
        value = self.value_head(combined_embeds)  # [batch_size, 1]
        return policy_logits, value
    
    def act(self, state):
        state = state.to(device)
        policy_logits, _ = self.forward(state)
        actions = []
        action_logprobs = []
        for logits in policy_logits:
            dist = Categorical(logits=logits)
            action = dist.sample()
            actions.append(action.item())
            action_logprob = dist.log_prob(action)
            action_logprobs.append(action_logprob)
        return np.array(actions), torch.stack(action_logprobs)

    def evaluate(self, state, actions):
        policy_logits, value = self.forward(state)
        action_logprobs = []
        dist_entropies = []
        for i, logits in enumerate(policy_logits):
            dist = Categorical(logits=logits)
            action_logprob = dist.log_prob(actions[:, i])
            dist_entropy = dist.entropy()
            action_logprobs.append(action_logprob)
            dist_entropies.append(dist_entropy)
        return torch.stack(action_logprobs, dim=1), value.squeeze(-1), torch.stack(dist_entropies, dim=1)

# 메모리 클래스 정의
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

# PPO 트레이너 클래스 정의
class PPOTrainer:
    def __init__(self, env, policy, memory, optimizer, gamma=0.99, epsilon=0.2, epochs=10, entropy_coef=0.05):
        self.env = env
        self.policy = policy
        self.memory = memory
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.entropy_coef = entropy_coef

    def train(self, max_episodes, model_name):
        best_reward = float('-inf')
        
        rows = len(env.real_dfs['000660'])
        
        data_slices = []
        for i in range(rows // 1000):
            data_slices.append((i*1000, (i+1)*1000))
            
        num_slices = len(data_slices)
        
        for episode in range(max_episodes):
            data_range = data_slices[episode % num_slices]
            state = self.env.reset(data_range=data_range)
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                actions, action_logprobs = self.policy.act(state_tensor)
                next_state, reward, done, _ = self.env.step(actions)
                total_reward += reward
                self.memory.states.append(state)
                self.memory.actions.append(actions)
                self.memory.logprobs.append(action_logprobs.sum().item())
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                state = next_state

            # 에피소드가 끝나면 PPO 업데이트
            self.ppo_update()
            self.memory.clear()

            # 결과 출력 및 시각화
            file_name = f'C:\\Users\\G-10\\Desktop\\final\\{model_name}\\plot\\plot_episode_{episode + 1}'
            self.plot_results(episode, total_reward, file_name)
            
                # 베스트 모델 저장
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(policy.state_dict(), f'C:\\Users\\G-10\\Desktop\\final\\{model_name}\\model\\best_ppo_actor_critic_model_{episode + 1}_{total_reward:.3f}.pth')
                print(f'Episode {episode + 1 }: New best model saved with total reward: {total_reward:.2f}')

    def ppo_update(self):
        if len(self.memory.states) == 0:
            return

        # 메모리에서 텐서로 변환
        states = torch.tensor(self.memory.states, dtype=torch.float32).to(device)
        actions = torch.tensor(self.memory.actions, dtype=torch.int64).to(device)
        old_logprobs = torch.tensor(self.memory.logprobs, dtype=torch.float32).to(device)
        rewards = self.memory.rewards
        is_terminals = self.memory.is_terminals

        # 리턴 계산
        returns = self.compute_returns(rewards, is_terminals)

        # 어드밴티지 계산
        advantages = self.compute_advantages(states, returns)

        # 정책 업데이트
        for epoch in range(self.epochs):
            action_logprobs, state_values, dist_entropies = self.policy.evaluate(states, actions)
            total_logprobs = action_logprobs.sum(dim=1)
            total_entropies = dist_entropies.sum(dim=1)
            ratios = torch.exp(total_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            value_loss = F.mse_loss(state_values.squeeze(-1), returns)
            loss = -torch.min(surr1, surr2) + 0.5 * value_loss - self.entropy_coef * total_entropies

            self.optimizer.zero_grad()
            loss_mean = loss.mean()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()  

            print(f"Epoch {epoch+1}, Loss: {loss_mean.item():.4f}")

    def compute_returns(self, rewards, is_terminals):
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        return torch.tensor(returns, dtype=torch.float32).to(device)

    def compute_advantages(self, states, returns):
        with torch.no_grad():
            _, state_values = self.policy.forward(states)
            advantages = returns - state_values.squeeze(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def plot_results(self, episode, total_reward, filename):
        # 행동 분포 및 결과 시각화
        action_counts = np.zeros(9)
        for actions in self.env.action_history:
            action_counts += np.bincount(actions, minlength=9)

        plt.figure(figsize=(18, 12))
        plt.subplot(4, 1, 1)
        plt.plot(self.env.portfolio_value_history)
        plt.title(f'Episode {episode+1} - Portfolio Value Over Time')
        plt.ylabel('Portfolio Value')
        
        
        plt.subplot(4, 1, 2)
        stock_price = self.env.price_history['000660']
        plt.plot(stock_price)
        plt.title('real_price')
        plt.ylabel('Number of Stocks')
        
        # 'sell' 행동에 대한 인덱스 찾기
        sell_indices = [i for i, actions in enumerate(self.env.action_history) if any(action in [0, 1, 2, 3] for action in actions)]
        buy_indices = [i for i, actions in enumerate(self.env.action_history) if any(action in [5, 6, 7, 8] for action in actions)]
        
        # 'sell' 행동이 있는 인덱스에 주석 추가
        for idx in sell_indices:
            plt.plot(idx, self.env.price_history['000660'][idx], 'o', color='red', markersize=2)  # 빨간 점 찍기
        for idx in buy_indices:
            plt.plot(idx, self.env.price_history['000660'][idx], 'o', color='black', markersize=2)  
        
        plt.legend()
        
        plt.subplot(4, 1, 3)
        action_labels = ['Sell 100%', 'Sell 75%', 'Sell 50%', 'Sell 25%', 'Hold', 
                        'Buy 25%', 'Buy 50%', 'Buy 75%', 'Buy 100%']
        plt.bar(action_labels, action_counts, color=['red', 'darkred', 'orange', 'lightcoral', 'gray', 
                                                    'lightgreen', 'green', 'darkgreen', 'lime'])
        plt.title('Action Distribution')
        plt.ylabel('Counts')
        plt.xticks(rotation=45)

        # 각 막대 위에 수치 추가
        for i, count in enumerate(action_counts):
            plt.text(i, count + 0.5, str(count), ha='center', fontsize=12)  # 0.5는 수치가 막대 위에 약간 떠 있도록 조정

        plt.tight_layout()  # 레이아웃 조정

        plt.subplot(4, 1, 4)
        initial_prices = {ticker: self.env.price_history[ticker][0] for ticker in tickers}
        final_prices = {ticker: self.env.price_history[ticker][-1] for ticker in tickers}
        returns = []
        for ticker in tickers:
            if initial_prices[ticker] == 0:
                ret = 0
            else:
                ret = (final_prices[ticker] - initial_prices[ticker]) / initial_prices[ticker] * 100
            returns.append(ret)
        plt.bar(tickers, returns, color=['blue', 'orange', 'purple'])
        plt.title('Stock Returns (%)')
        plt.ylabel('Return (%)')


        plt.tight_layout()
        plt.savefig(filename)  # 파일로 저장
        plt.close()  # 그래프를 닫아 메모리 해제


        print(f"Episode {episode+1} completed. Total Reward: {total_reward:.2f}, Final Portfolio Value: {self.env.portfolio_value_history[-1]:.2f}")
        print(f"Action counts:")
        for i, label in enumerate(action_labels):
            print(f"  {label}: {int(action_counts[i])}")
        print('-' * 50)

# 데이터 로드 및 환경 설정
# tickers = ['000660', '005930', '042700']
tickers = ['000660']
dfs = {}

# 데이터 로드 및 기술적 지표 계산
for ticker in tickers:
    # df = yf.download(ticker, start='2019-01-01', end='2023-12-31', progress=False)  # 데이터 기간을 5년으로 확장
    df = pd.read_csv(f'{ticker}.csv', encoding='cp949')
    # df = df[160000:161000]
    df = df.rename(columns={'Unnamed: 0': 'Time', 
                                '매수량': 'BuyVolume', 
                                '매도량': 'SellVolume', 
                                '종가': 'Close', 
                                '저가': 'Low',
                                '시가': 'Open',
                                '고가': 'High',
                                '거래량': 'Volume'})
    # df['Power'] = df['BuyVolume'] - df['SellVolume']
    
    # 원래의 Close 가격 보존
    df.loc[:, 'Close_unscaled'] = df['Close']  # 실제 가격 저장

    # 기술적 지표 계산
    df.loc[:, 'MA10'] = df['Close'].rolling(window=10).mean()

    # 볼린저 밴드 계산 (20일 이동평균 ± 2표준편차)
    # df.loc[:, 'Upper_Band'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
    # df.loc[:, 'Lower_Band'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()

    # RSI 계산
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
    
    columns = ['Power', 'Close', 'Low', 'Open', 'High', 'MA10', 'MA20', 'MA60', 'MA120', 'Upper_Band', 'Lower_Band']
    
    # 변화율 데이터 추가
    # df[columns] = df[columns].pct_change() * 100
    df = df.dropna().copy()  # 복사하여 경고 방지
    
    # 입력 데이터 정규화 (거래량과 볼린저밴드 포함)
    standard_scaler = StandardScaler()
    # robust_scaler = RobustScaler()
    
    feature_cols = ['Close', 'Low', 'MA10', 'MA20', 'RSI', 'Upper_Band', 'Lower_Band', 'Volume']
    # power_cols = ['Power']
    df.loc[:, feature_cols] = standard_scaler.fit_transform(df[feature_cols])
    # df.loc[:, power_cols] = robust_scaler.fit_transform(df[power_cols])

    dfs[ticker] = df.reset_index(drop=True)

# 환경 생성
env = MultiStockTradingEnv(dfs, stock_dim=len(tickers), initial_balance=100000000, seq_length=20, input_dim=len(feature_cols))

# PPO 모델, 옵티마이저 및 메모리 초기화
policy = ActorCritic([9] * env.stock_dim, seq_length=env.seq_length, input_dim=env.input_dim).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-5)  # 학습률 감소
memory = Memory()

# 트레이너 생성
trainer = PPOTrainer(env, policy, memory, optimizer, gamma=0.6, epsilon=0.5, epochs=15)

model_name = 'model_22'
# 학습 시작
trainer.train(max_episodes=3000, model_name=model_name)
