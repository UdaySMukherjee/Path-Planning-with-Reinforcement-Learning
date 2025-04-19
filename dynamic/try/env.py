import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 11
CELL_SIZE = 10
AGENT_SPEED = CELL_SIZE

class StaticEnvironment:
    def __init__(self, initial_position, target_position):
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.max_coord = (GRID_SIZE - 1) * CELL_SIZE
        self.initial_state = np.asarray(initial_position, dtype=float)
        self.goal_state = np.asarray(target_position, dtype=float)
        self.agent_state = np.copy(self.initial_state)
        self.vision_range = 2 * CELL_SIZE

        self.action_space = {
            0: [AGENT_SPEED, 0], 1: [0, AGENT_SPEED],
            2: [-AGENT_SPEED, 0], 3: [0, -AGENT_SPEED],
            4: [-AGENT_SPEED, AGENT_SPEED], 5: [-AGENT_SPEED, -AGENT_SPEED],
            6: [AGENT_SPEED, AGENT_SPEED], 7: [AGENT_SPEED, -AGENT_SPEED]
        }
        self.num_actions = len(self.action_space)
        self.steps_counter = 0
        self.max_episode_steps = 200

    def reset(self):
        self.agent_state = np.copy(self.initial_state)
        self.steps_counter = 0
        return self.agent_state

    def step(self, action):
        move = self.action_space[action]
        new_pos = self.agent_state + np.asarray(move)
        new_pos = np.clip(new_pos, 0, self.max_coord)

        self.agent_state = new_pos
        done = self._is_goal()
        reward = 100 if done else -1

        self.steps_counter += 1
        if self.steps_counter >= self.max_episode_steps:
            done = True

        return self.agent_state, reward, done

    def _is_goal(self):
        return np.linalg.norm(self.agent_state - self.goal_state) < (AGENT_SPEED / 2)

    def render_path(self, path, label):
        xs = [state[0] for state in path]
        ys = [state[1] for state in path]
        plt.plot(xs, ys, marker='o', label=label)

    def visualize(self, paths_dict):
        plt.figure(figsize=(6, 6))
        plt.grid(True)
        for agent, path in paths_dict.items():
            self.render_path(path, label=agent)
        plt.scatter(*self.initial_state, color='green', s=100, label='Start')
        plt.scatter(*self.goal_state, color='red', s=100, label='Goal')
        plt.title("Agent Paths")
        plt.legend()
        plt.xlim(0, self.max_coord)
        plt.ylim(0, self.max_coord)
        plt.show()
