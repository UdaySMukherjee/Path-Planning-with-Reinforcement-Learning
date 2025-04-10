import numpy as np
import random
import torch
import torch.nn as nn  # <-- Add this line
import torch.optim as optim
from collections import deque
from env import DynamicEnvironment, final_states
import matplotlib.pyplot as plt
import time


# Constants from our DynamicEnvironment
GRID_SIZE = 11
CELL_SIZE = 10
X_MAX_COORD = (GRID_SIZE - 1) * CELL_SIZE
Y_MAX_COORD = (GRID_SIZE - 1) * CELL_SIZE

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Adjusted decay
        self.learning_rate = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model().to(self.device).double()  # Double precision
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def build_model(self):
        """Create a neural network for approximating Q-values."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience for experience replay."""
        self.memory.append((np.array(state, dtype=np.float64),
                            action,
                            reward,
                            np.array(next_state, dtype=np.float64),
                            done))

    def act(self, state):
        """Return action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(np.array(state, dtype=np.float64)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values[0]).item()

    def replay(self, batch_size):
        """Train the model using random experiences from the memory."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in minibatch])).to(self.device)
        actions = torch.tensor([e[1] for e in minibatch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e[2] for e in minibatch], dtype=torch.double).to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch])).to(self.device)
        dones = torch.tensor([e[4] for e in minibatch], dtype=torch.bool).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_dqn_episodes(env, agent, episodes=1000, batch_size=64):
    rewards_per_episode = []
    steps_per_episode = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))

        while not done:
            ax.clear()
            ax.set_xlim(-CELL_SIZE/2, X_MAX_COORD + CELL_SIZE/2)
            ax.set_ylim(-CELL_SIZE/2, Y_MAX_COORD + CELL_SIZE/2)
            ax.set_xticks(np.arange(0, X_MAX_COORD + CELL_SIZE, CELL_SIZE))
            ax.set_yticks(np.arange(0, Y_MAX_COORD + CELL_SIZE, CELL_SIZE))
            ax.grid(True)

            for (ox, oy, _, _) in env.obstacles:
                ax.add_patch(plt.Rectangle((ox - CELL_SIZE/2, oy - CELL_SIZE/2), CELL_SIZE, CELL_SIZE, color='gray'))
            for (ox, oy) in env.static_obstacles:
                ax.add_patch(plt.Rectangle((ox - CELL_SIZE/2, oy - CELL_SIZE/2), CELL_SIZE, CELL_SIZE, color='black'))

            ax.scatter(env.vector_initial_state[0], env.vector_initial_state[1], color='green', s=150, label='Start', zorder=5)
            ax.scatter(env.vector_terminal_state[0], env.vector_terminal_state[1], color='red', s=150, label='Terminal', zorder=5)
            ax.scatter(env.vector_agent_state[0], env.vector_agent_state[1], color='blue', s=100, label='Agent', zorder=5)
            ax.set_title(f"Episode: {e + 1}, Step: {steps}, Epsilon: {agent.epsilon:.2f}")
            plt.draw()
            plt.pause(0.01)

            action = agent.act(state)
            next_state, next_state_flag, reward, done, _ = env.step(action)

            total_reward += reward
            steps += 1
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(batch_size)

            if done:
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        plt.ioff()
        plt.close(fig)

        print(f"Episode {e + 1}/{episodes}, Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.2f}")

    return rewards_per_episode, steps_per_episode

if __name__ == "__main__":
    start_pos = [0.0, 0.0]
    end_pos = [X_MAX_COORD, Y_MAX_COORD]
    env = DynamicEnvironment(initial_position=start_pos, target_position=end_pos)

    state_size = 2
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)

    rewards, steps = run_dqn_episodes(env, agent, episodes=50, batch_size=64)

    # Save training results
    print("Training finished.")
    plt.plot(rewards)
    plt.title('DQN Training Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.show()
