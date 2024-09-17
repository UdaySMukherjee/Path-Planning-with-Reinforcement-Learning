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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
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

    def remember(self, state, action, reward, next_state, done):
        """Store experience for experience replay."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Return action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        """Train the model using random experiences from the memory."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state))
            target_f = self.model(state).clone().detach()  # Detach to avoid in-place modification issue
            target_f[action] = target
            output = self.model(state)
            loss = self.criterion(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def state_to_vector(state, size):
    """Flatten the 2D grid state to a 1D vector."""
    vector = np.zeros(size)
    vector[state[0] * int(np.sqrt(size)) + state[1]] = 1
    return vector

# Inside DQN.py

def get_next_state(env, state, action):
    """Simulate the next state based on the current state and the action taken."""
    row, col = state
    move = DIRECTIONS[action]
    next_row, next_col = row + move[0], col + move[1]
    if 0 <= next_row < env.shape[0] and 0 <= next_col < env.shape[1] and env[next_row, next_col] != 1:
        return (next_row, next_col)
    return state  # If invalid move, stay in the same position

def run_dqn_episodes(env, agent, episodes=100, batch_size=32):
    """Train the DQN agent over multiple episodes."""
    rewards_per_episode = []
    steps_per_episode = []
    best_path = []
    start_time = time.time()

    max_steps_per_episode = env.shape[0] * env.shape[1]  # Limit steps per episode

    for e in range(episodes):
        state = (0, 0)
        total_reward = 0
        steps = 0
        done = False
        path_dqn = [state]

        while not done:
            state_vector = state_to_vector(state, env.size)
            action = agent.act(state_vector)
            next_state = get_next_state(env, state, action)
            reward = -1  # Penalty for each step

            if next_state == state:
                reward -= 10  # Penalty for hitting an obstacle
            if next_state == (env.shape[0] - 1, env.shape[1] - 1):
                reward += 100  # High reward for reaching the goal
                done = True

            total_reward += reward
            steps += 1

            next_state_vector = state_to_vector(next_state, env.size)
            agent.remember(state_vector, action, reward, next_state_vector, done)

            state = next_state
            path_dqn.append(state)

            if done or steps >= max_steps_per_episode:
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if path_dqn[-1] == (env.shape[0] - 1, env.shape[1] - 1):
            best_path = path_dqn

        print(f"Episode: {e + 1}/{episodes}, Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.2f}")

    end_time = time.time()
    execution_time = end_time - start_time
    path_length = len(best_path) if best_path else float('inf')

    return rewards_per_episode, steps_per_episode, best_path, execution_time, path_length


def visualize_path(env, path, filename="path_dqn.png"):
    """Visualize the environment and the path using a line and save as an image"""
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
    plt.title("Path Found by DQN")
    
    # Save the plot as an image
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    size = random.choice([10, 15, 20, 25, 30])
    env = generate_env(40)
    visualize_env(env)  # Show the environment

    agent = DQNAgent(state_size=env.size, action_size=len(DIRECTIONS))

    # Run DQN agent for 100 episodes
    rewards, steps, best_path, exec_time, path_length = run_dqn_episodes(env, agent, episodes=100)

    if best_path:
        print(f"Path found! Length: {path_length} steps")
        print(f"Execution time: {exec_time:.4f} seconds")

        # Visualize the best path found and save the path plot as an image
        visualize_path(env, best_path, filename="path_dqn.png")
    else:
        print("No Path Found")
