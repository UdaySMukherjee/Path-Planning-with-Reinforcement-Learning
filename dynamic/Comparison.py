import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from env import DynamicEnvironment, final_states

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.alpha = 0.01
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)
        target = reward + self.gamma * torch.max(next_q_values).item()
        target_f = q_values.clone()
        target_f[0][action] = target
        loss = self.criterion(q_values, target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# SARSA Agent
class SARSAAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.alpha = 0.01
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, next_action):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)
        target = reward + self.gamma * next_q_values[0][next_action].item()
        target_f = q_values.clone()
        target_f[0][action] = target
        loss = self.criterion(q_values, target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.memory = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 5000:
            self.memory.pop(0)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            loss = self.criterion(target_f, self.model(state_tensor))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Evaluation & Visualization
def evaluate_agent(rewards):
    rewards = np.array(rewards)
    successes = np.sum(rewards > 0)
    failures = len(rewards) - successes
    metrics = {
        "Total Episodes": len(rewards),
        "Average Reward": np.mean(rewards),
        "Max Reward": np.max(rewards),
        "Min Reward": np.min(rewards),
        "Success Rate (reward > 0)": successes / len(rewards),
        "Success Count": successes,
        "Failure Count": failures
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    return metrics

def compare_algorithms_plot(results_dict):
    plt.figure(figsize=(12, 6))
    for name, rewards in results_dict.items():
        plt.plot(rewards, label=f"{name} (Avg: {np.mean(rewards):.2f})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Comparison of RL Algorithms on Dynamic Environment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Training Loops
def run_qlearning_episodes(env, agent, episodes=100):
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, _, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return rewards

def run_sarsa_episodes(env, agent, episodes=100):
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        while not done:
            next_state, _, reward, done, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.learn(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            total_reward += reward
        rewards.append(total_reward)
    return rewards

def run_dqn_episodes(env, agent, episodes=100):
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, _, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return rewards

# Run Everything
if __name__ == "__main__":
    env1 = DynamicEnvironment([0.0, 0.0], [100, 100])
    q_agent = QLearningAgent(2, env1.num_actions)
    rewards_q = run_qlearning_episodes(env1, q_agent, episodes=50)
    print("\nQ-Learning Evaluation:")
    evaluate_agent(rewards_q)

    env2 = DynamicEnvironment([0.0, 0.0], [100, 100])
    s_agent = SARSAAgent(2, env2.num_actions)
    rewards_s = run_sarsa_episodes(env2, s_agent, episodes=50)
    print("\nSARSA Evaluation:")
    evaluate_agent(rewards_s)

    env3 = DynamicEnvironment([0.0, 0.0], [100, 100])
    dqn_agent = DQNAgent(2, env3.num_actions)
    rewards_d = run_dqn_episodes(env3, dqn_agent, episodes=50)
    print("\nDQN Evaluation:")
    evaluate_agent(rewards_d)

    compare_algorithms_plot({
        "Q-Learning": rewards_q,
        "SARSA": rewards_s,
        "DQN": rewards_d
    })
