import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from collections import deque
from env import generate_env, visualize_env

# Define 4 possible movements: right, down, left, up
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # Discount factor
        self.alpha = 0.01  # Learning rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

    def build_model(self):
        """Create a neural network for approximating Q-values."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def choose_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0))  # Add batch dimension
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state):
        """Update the Q-values using Q-learning update rule."""
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)

        q_values = self.model(state.unsqueeze(0))  # Add batch dimension
        next_q_values = self.model(next_state.unsqueeze(0))  # Add batch dimension

        target = reward + self.gamma * torch.max(next_q_values).item()
        target_f = q_values.clone()
        target_f[0][action] = target

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_f)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def state_to_vector(state, size):
    """Convert a 2D state to a 1D vector."""
    vector = np.zeros(size)
    vector[state[0] * int(np.sqrt(size)) + state[1]] = 1
    return vector

def get_next_state(env, state, action):
    """Simulate the next state based on the current state and the action taken."""
    row, col = state
    move = DIRECTIONS[action]
    next_row, next_col = row + move[0], col + move[1]
    if 0 <= next_row < env.shape[0] and 0 <= next_col < env.shape[1] and env[next_row, next_col] != 1:
        return (next_row, next_col)
    return state  # If invalid move, stay in the same position

# Inside Q_learning.py

def run_qlearning_episodes(env, agent, episodes=100):
    """Train the Q-learning agent over multiple episodes."""
    rewards_per_episode = []
    steps_per_episode = []
    best_path = []
    highest_reward = float('-inf')
    start_time = time.time()

    max_steps_per_episode = env.shape[0] * env.shape[1]  # Limit steps per episode

    for e in range(episodes):
        state = (0, 0)
        total_reward = 0
        steps = 0
        done = False
        path_qlearning = [state]

        while not done:
            action = agent.choose_action(state_to_vector(state, env.size))
            next_state = get_next_state(env, state, action)
            reward = -1  # Penalty for each step

            if next_state == state:
                reward -= 10  # Penalty for hitting an obstacle
            if next_state == (env.shape[0] - 1, env.shape[1] - 1):
                reward += 100  # High reward for reaching the goal
                done = True

            total_reward += reward
            steps += 1

            agent.learn(state_to_vector(state, env.size), action, reward, state_to_vector(next_state, env.size))

            state = next_state
            path_qlearning.append(state)

            if done or steps >= max_steps_per_episode:
                break

        # Update best path if total_reward is higher
        if total_reward > highest_reward:
            highest_reward = total_reward
            best_path = path_qlearning

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        print(f"Episode: {e + 1}/{episodes}, Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.2f}")

    end_time = time.time()
    execution_time = end_time - start_time
    path_length = len(best_path) if best_path else float('inf')

    return rewards_per_episode, steps_per_episode, best_path, execution_time, path_length


def plot_best_path(env, path, filename="best_path_qlearning.png"):
    """Plot the best path found by the agent."""
    plt.imshow(env, cmap='gray_r', origin='upper')
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, 'b-o', markersize=4, linewidth=2, label='Q-Learning Path')
    plt.title('Best Path Found by Q-Learning')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.legend()
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    size = random.choice([10, 15, 20, 25, 30])
    env = generate_env(40)
    agent = QLearningAgent(state_size=env.size, action_size=len(DIRECTIONS))
    rewards, steps, best_path, exec_time, path_length = run_qlearning_episodes(env, agent)
    print(f"Execution Time: {exec_time} seconds")
    print(f"Path Length: {path_length}")
    if best_path:
        print(f"Best Path: {best_path}")
        plot_best_path(env, best_path)
    else:
        print("No path found.")
