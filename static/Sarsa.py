import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from env import generate_env

# Define 4 possible movements: right, down, left, up
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

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

# Inside Sarsa.py

def run_sarsa_episodes(env, agent, episodes=100):
    """Train the SARSA agent over multiple episodes."""
    best_path = []
    highest_reward = float('-inf')
    start_time = time.time()

    max_steps_per_episode = env.shape[0] * env.shape[1]  # Limit steps per episode

    for e in range(episodes):
        state = (0, 0)
        total_reward = 0
        steps = 0
        done = False
        path_sarsa = [state]

        action = agent.choose_action(state_to_vector(state, env.size))
        
        while not done:
            next_state = get_next_state(env, state, action)
            reward = -1  # Penalty for each step

            if next_state == state:
                reward -= 10  # Penalty for hitting an obstacle
            if next_state == (env.shape[0] - 1, env.shape[1] - 1):
                reward += 100  # High reward for reaching the goal
                done = True

            total_reward += reward
            steps += 1

            next_action = agent.choose_action(state_to_vector(next_state, env.size))
            agent.learn(state_to_vector(state, env.size), action, reward, state_to_vector(next_state, env.size), next_action)

            state = next_state
            action = next_action
            path_sarsa.append(state)

            if done or steps >= max_steps_per_episode:
                break

        # Update best path if total_reward is higher
        if total_reward > highest_reward:
            highest_reward = total_reward
            best_path = path_sarsa

        print(f"Episode: {e + 1}/{episodes}, Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.2f}")

    execution_time = time.time() - start_time
    path_length = len(best_path) if best_path else float('inf')

    return best_path, execution_time, path_length


def plot_best_path(env, path, filename="best_path_sarsa.png"):
    """Visualize the environment and the best path using a line and save as an image"""
    cmap = plt.cm.get_cmap('tab20c', 4)  # Custom color map
    plt.imshow(env, cmap=cmap)

    # Extract x and y coordinates from the path
    x_coords = [pos[1] for pos in path]
    y_coords = [pos[0] for pos in path]

    # Plot the path using lines
    plt.plot(x_coords, y_coords, color="blue", marker="o", linestyle="-", linewidth=2, markersize=4)

    # Mark start and end points
    plt.scatter([0], [0], color="green", label="Start", s=100)
    plt.scatter([env.shape[1] - 1], [env.shape[0] - 1], color="red", label="End", s=100)

    plt.grid(True, color="black", linewidth=0.5)
    plt.title("Best Path Found by SARSA")
    
    # Save the plot as an image
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Generate environment
    size = random.choice([10, 15, 20, 25, 30])
    env = generate_env(40)

    # Create SARSA agent
    agent = SARSAAgent(state_size=env.size, action_size=len(DIRECTIONS))

    # Run SARSA agent for 100 episodes
    best_path, exec_time, path_length = run_sarsa_episodes(env, agent, episodes=100)

    # Check and save the best path
    if best_path:
        print(f"Best Path Length: {path_length} steps")
        print(f"Execution Time: {exec_time:.4f} seconds")
        plot_best_path(env, best_path, filename="best_path_sarsa.png")
    else:
        print("No path found.")
