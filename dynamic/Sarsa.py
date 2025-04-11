import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from env import DynamicEnvironment, final_states

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

def run_sarsa_episodes(env, agent, episodes=100):
    rewards_per_episode = []
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        step_count = 0
        done = False

        while not done:
            ax.clear()
            ax.set_xlim(-env.cell_size/2, env.x_max_coord + env.cell_size/2)
            ax.set_ylim(-env.cell_size/2, env.y_max_coord + env.cell_size/2)
            ax.set_xticks(np.arange(0, env.x_max_coord + env.cell_size, env.cell_size))
            ax.set_yticks(np.arange(0, env.y_max_coord + env.cell_size, env.cell_size))
            ax.grid(True)

            for ox, oy, *_ in env.obstacles:
                ax.add_patch(plt.Rectangle((ox - env.cell_size/2, oy - env.cell_size/2), env.cell_size, env.cell_size, color='gray'))
            for ox, oy in env.static_obstacles:
                ax.add_patch(plt.Rectangle((ox - env.cell_size/2, oy - env.cell_size/2), env.cell_size, env.cell_size, color='black'))

            ax.scatter(*env.vector_initial_state, color='green', s=150, label='Start')
            ax.scatter(*env.vector_terminal_state, color='red', s=150, label='End')
            ax.scatter(*env.vector_agent_state, color='blue', s=100, label='Agent')
            ax.set_title(f"SARSA - Ep {ep+1}, Step {step_count}, Eps {agent.epsilon:.2f}")
            plt.draw()
            plt.pause(0.01)

            next_state, _, reward, done, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.learn(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action
            total_reward += reward
            step_count += 1

        rewards_per_episode.append(total_reward)
        print(f"Episode {ep+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    plt.ioff()
    plt.close(fig)
    return rewards_per_episode

if __name__ == "__main__":
    env = DynamicEnvironment(initial_position=[0.0, 0.0], target_position=[100, 100])
    agent = SARSAAgent(state_size=2, action_size=env.num_actions)
    rewards = run_sarsa_episodes(env, agent, episodes=50)
    visualize_training_progress(env, rewards, "SARSA", episode_num="final_sarsa")
