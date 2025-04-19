import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
# from env1 import StaticEnvironment
from env2 import DynamicEnvironment

class VisionMixin:
    def get_safe_actions(self, state):
        safe_actions = []
        for a, move in self.env.action_space.items():
            new_pos = state + np.array(move)
            new_pos = np.clip(new_pos, 0, self.env.max_coord)
            if np.linalg.norm(new_pos - state) <= self.env.vision_range:
                safe_actions.append(a)
        return safe_actions if safe_actions else list(self.env.action_space.keys())

class QLearningAgent(VisionMixin):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.path = []
        self.rewards = []

    def state_to_key(self, state):
        return tuple((state / self.env.cell_size).astype(int))

    def run_episode(self):
        state = self.env.reset()
        self.path = [state.copy()]
        total_reward = 0

        for _ in range(self.env.max_episode_steps):
            key = self.state_to_key(state)
            if key not in self.q_table:
                self.q_table[key] = np.zeros(self.env.num_actions)

            if random.random() < self.epsilon:
                action = random.choice(self.get_safe_actions(state))
            else:
                safe = self.get_safe_actions(state)
                q_vals = self.q_table[key][safe]
                action = safe[np.argmax(q_vals)]

            next_state, _, reward, done, _ = self.env.step(action)
            # next_state, reward, done = self.env.step(action)   #for static
            next_key = self.state_to_key(next_state)

            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(self.env.num_actions)

            self.q_table[key][action] += self.alpha * (
                reward + self.gamma * np.max(self.q_table[next_key]) - self.q_table[key][action]
            )

            self.path.append(next_state.copy())
            state = next_state
            total_reward += reward
            if done:
                break

        self.rewards.append(total_reward)
        return total_reward

class SARSAAgent(QLearningAgent):
    def run_episode(self):
        state = self.env.reset()
        self.path = [state.copy()]
        total_reward = 0
        key = self.state_to_key(state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.env.num_actions)
        action = random.choice(self.get_safe_actions(state)) if random.random() < self.epsilon else np.argmax(self.q_table[key])

        for _ in range(self.env.max_episode_steps):
            next_state, _, reward, done, _ = self.env.step(action)
            # next_state, reward, done = self.env.step(action)   #static
            next_key = self.state_to_key(next_state)

            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(self.env.num_actions)

            next_action = random.choice(self.get_safe_actions(next_state)) if random.random() < self.epsilon else np.argmax(self.q_table[next_key])

            self.q_table[key][action] += self.alpha * (
                reward + self.gamma * self.q_table[next_key][next_action] - self.q_table[key][action]
            )

            self.path.append(next_state.copy())
            state, key, action = next_state, next_key, next_action
            total_reward += reward
            if done:
                break

        self.rewards.append(total_reward)
        return total_reward

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent(VisionMixin):
    def __init__(self, env, gamma=0.9, epsilon=0.1, lr=0.001):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = DQN(2, env.num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.path = []
        self.rewards = []

    def run_episode(self):
        state = self.env.reset()
        self.path = [state.copy()]
        total_reward = 0

        for _ in range(self.env.max_episode_steps):
            # state_tensor = torch.FloatTensor(state / self.env.max_coord)
            state_tensor = torch.FloatTensor(state / self.env.max_coord).cuda()

            q_values = self.model(state_tensor)
            safe_actions = self.get_safe_actions(state)

            if random.random() < self.epsilon:
                action = random.choice(safe_actions)
            else:
                # q_safe = q_values[safe_actions].detach().numpy()
                q_safe = q_values[safe_actions].detach().cpu().numpy()
                action = safe_actions[np.argmax(q_safe)]

            # next_state, reward, done = self.env.step(action)   #static
            next_state, _, reward, done, _ = self.env.step(action)
            # next_q = self.model(torch.FloatTensor(next_state / self.env.max_coord)).detach()
            next_q = self.model(torch.FloatTensor(next_state / self.env.max_coord).cuda()).detach()


            target = q_values.clone().detach()
            target[action] = reward + self.gamma * torch.max(next_q)

            loss = self.criterion(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.path.append(next_state.copy())
            total_reward += reward
            state = next_state
            if done:
                break

        self.rewards.append(total_reward)
        return total_reward

def evaluate_agent(agent_class, env, episodes=50):
    agent = agent_class(env)
    successes = 0
    failures = 0
    rewards = []

    for _ in range(episodes):
        reward = agent.run_episode()
        rewards.append(reward)
        if reward > 0:
            successes += 1
        else:
            failures += 1

    stats = {
        "Total Episodes": len(rewards),
        "Average Reward": np.mean(rewards),
        "Max Reward": np.max(rewards),
        "Min Reward": np.min(rewards),
        "Success Rate (reward > 0)": successes / len(rewards),
        "Success Count": successes,
        "Failure Count": failures
    }

    return agent.path, stats, rewards
'''
def main():
    # env = StaticEnvironment([0, 0], [100, 100])
    env = DynamicEnvironment([0, 0], [100, 100])
    paths = {}

    print("Evaluating Q-Learning...")
    q_path, q_stats, q_rewards = evaluate_agent(QLearningAgent, env)
    print(q_stats)
    paths['Q-Learning'] = q_path

    print("Evaluating SARSA...")
    sarsa_path, sarsa_stats, sarsa_rewards = evaluate_agent(SARSAAgent, env)
    print(sarsa_stats)
    paths['SARSA'] = sarsa_path

    print("Evaluating DQN...")
    dqn_path, dqn_stats, dqn_rewards = evaluate_agent(DQNAgent, env)
    print(dqn_stats)
    paths['DQN'] = dqn_path

    # Plotting Reward vs Epoch for each agent
    plt.figure(figsize=(10, 6))
    plt.plot(q_rewards, label="Q-Learning")
    plt.plot(sarsa_rewards, label="SARSA")
    plt.plot(dqn_rewards, label="DQN")
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize paths for each agent
    # env.visualize(paths)  #static
    env.render()


if __name__ == "__main__":
    main()
'''

import matplotlib.pyplot as plt

def evaluate_agents_with_live_visualization(env, episodes=100):
    q_agent = QLearningAgent(env)
    sarsa_agent = SARSAAgent(env)
    dqn_agent = DQNAgent(env)
    dqn_agent.model = dqn_agent.model.cuda()  # Move model to GPU

    q_rewards, sarsa_rewards, dqn_rewards = [], [], []
    q_successes, sarsa_successes, dqn_successes = 0, 0, 0

    # Set up reusable figure and axes
    plt.ion()  # Turn on interactive mode
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.show()

    for ep in range(episodes):
        print(f"\nEpisode {ep + 1}/{episodes}")

        # Reset env
        q_agent.env.reset()
        sarsa_agent.env.reset()
        dqn_agent.env.reset()

        # Run episodes
        q_agent.run_episode()
        sarsa_agent.run_episode()
        dqn_agent.run_episode()

        # Store rewards
        q_rewards.append(q_agent.rewards[-1])
        sarsa_rewards.append(sarsa_agent.rewards[-1])
        dqn_rewards.append(dqn_agent.rewards[-1])

        if q_agent.rewards[-1] > 0:
            q_successes += 1
        if sarsa_agent.rewards[-1] > 0:
            sarsa_successes += 1
        if dqn_agent.rewards[-1] > 0:
            dqn_successes += 1

        # Clear axes for reuse
        for ax in axs:
            ax.clear()

        for ax, agent, title in zip(axs, [q_agent, sarsa_agent, dqn_agent], ["Q-Learning", "SARSA", "DQN"]):
            ax.set_title(f"{title} - Ep {ep+1}")
            ax.set_xlim(0, env.x_max_coord)
            ax.set_ylim(0, env.y_max_coord)
            ax.grid(True)

            # Obstacles
            for x, y in env.static_obstacles:
                ax.add_patch(plt.Rectangle((x - 5, y - 5), 10, 10, color='black'))
            for x, y, *_ in env.obstacles:
                ax.add_patch(plt.Rectangle((x - 5, y - 5), 10, 10, color='gray'))

            # Start and goal
            sx, sy = env.vector_initial_state
            gx, gy = env.vector_terminal_state
            ax.add_patch(plt.Rectangle((sx - 5, sy - 5), 10, 10, color='green'))
            ax.add_patch(plt.Rectangle((gx - 5, gy - 5), 10, 10, color='red'))

            # Agent path
            xs = [p[0] for p in agent.path]
            ys = [p[1] for p in agent.path]
            ax.plot(xs, ys, marker='o', color='blue')

        plt.pause(0.1)

    plt.ioff()
    plt.close(fig)

    # Metrics calculation
    def get_metrics(rewards, successes):
        return {
            "Total Episodes": len(rewards),
            "Average Reward": np.mean(rewards),
            "Max Reward": np.max(rewards),
            "Min Reward": np.min(rewards),
            "Success Rate (reward > 0)": successes / len(rewards),
            "Success Count": successes,
            "Failure Count": len(rewards) - successes
        }

    q_metrics = get_metrics(q_rewards, q_successes)
    sarsa_metrics = get_metrics(sarsa_rewards, sarsa_successes)
    dqn_metrics = get_metrics(dqn_rewards, dqn_successes)

    print("\nFinal Metrics:")
    print("Q-Learning:", q_metrics)
    print("SARSA:", sarsa_metrics)
    print("DQN:", dqn_metrics)

    return q_rewards, sarsa_rewards, dqn_rewards



def main():
    env = DynamicEnvironment([0, 0], [100, 100])
    q_rewards, sarsa_rewards, dqn_rewards = evaluate_agents_with_live_visualization(env, episodes=100)

    # Final plot: reward comparison
    plt.figure(figsize=(10, 6))
    plt.plot(q_rewards, label="Q-Learning")
    plt.plot(sarsa_rewards, label="SARSA")
    plt.plot(dqn_rewards, label="DQN")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
