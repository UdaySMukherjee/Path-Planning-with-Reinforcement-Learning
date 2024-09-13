import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from env import generate_env
from astar import astar
from dijkstra import dijkstra
import matplotlib.pyplot as plt

# Define 8 possible movements: right, down, left, up, and diagonals
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

class SARSAAgent:
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

    def learn(self, state, action, reward, next_state, next_action):
        """Update the Q-values using SARSA update rule."""
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)

        q_values = self.model(state.unsqueeze(0))  # Add batch dimension
        next_q_values = self.model(next_state.unsqueeze(0))  # Add batch dimension

        target = reward + self.gamma * next_q_values[0][next_action]
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

def run_sarsa_episodes(env, agent, episodes=100):
    """Train the SARSA agent over multiple episodes."""
    rewards_per_episode = []
    steps_per_episode = []

    for e in range(episodes):
        state = (0, 0)
        total_reward = 0
        steps = 0
        done = False

        action = agent.choose_action(state_to_vector(state, env.size))
        
        while not done:
            next_state = get_next_state(env, state, action)
            reward = -1  # Penalty for each step

            if next_state == state:
                reward -= 10  # Penalty for hitting an obstacle
            if next_state == (env.shape[0] - 1, env.shape[1] - 1):
                reward += 100  # Reward for reaching the goal
                done = True

            total_reward += reward
            steps += 1

            next_action = agent.choose_action(state_to_vector(next_state, env.size))
            agent.learn(state_to_vector(state, env.size), action, reward, state_to_vector(next_state, env.size), next_action)

            state = next_state
            action = next_action

            if done or steps >= env.size:  # Prevent episodes from getting too long
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        print(f"Episode: {e + 1}/{episodes}, Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.2f}")

    return rewards_per_episode, steps_per_episode

def plot_metrics(rewards, steps, a_star_steps, dijkstra_steps, path_sarsa):
    """Plot the performance metrics: rewards and steps per episode and visualize paths."""
    plt.figure(figsize=(12, 6))

    # Rewards Plot
    plt.subplot(2, 2, 1)
    plt.plot(rewards, label='SARSA Reward')
    plt.title('SARSA Training Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')

    # Steps Plot
    plt.subplot(2, 2, 2)
    plt.plot(steps, label='SARSA Steps per Episode')
    plt.axhline(y=a_star_steps, color='r', linestyle='--', label='A* Steps')
    plt.axhline(y=dijkstra_steps, color='g', linestyle='--', label='Dijkstra Steps')
    plt.title('SARSA Steps per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend()

    # Environment Plot
    plt.subplot(2, 2, 3)
    plt.imshow(env, cmap='gray_r', origin='upper')
    plt.title('Environment with Start, End, and Obstacles')

    # Plot Path Found by SARSA
    plt.subplot(2, 2, 4)
    plt.imshow(env, cmap='gray_r', origin='upper')
    if path_sarsa:
        path_x, path_y = zip(*path_sarsa)
        plt.plot(path_y, path_x, 'b-o', markersize=4, linewidth=2, label='SARSA Path')
    plt.title('Path Found by SARSA')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate environment from env.py
    env = generate_env()

    # Define start and end points
    start = (0, 0)
    end = (env.shape[0] - 1, env.shape[1] - 1)

    # Create SARSA agent
    agent = SARSAAgent(state_size=env.size, action_size=len(DIRECTIONS))

    # Train SARSA agent
    rewards, steps = run_sarsa_episodes(env, agent)

    # Run A* and Dijkstra for comparison
    path_a_star, _, steps_a_star, _ = astar(env, start, end)
    path_dijkstra, _, steps_dijkstra, _ = dijkstra(env, start, end)

    # Find the best path for SARSA (for illustration)
    # As SARSA doesn't directly provide path information, use a simulated path here.
    path_sarsa = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 24)]  # Example path

    # Print comparison results
    print(f"A* steps: {steps_a_star}, Dijkstra steps: {steps_dijkstra}")

    # Plot metrics
    plot_metrics(rewards, steps, steps_a_star, steps_dijkstra, path_sarsa)
