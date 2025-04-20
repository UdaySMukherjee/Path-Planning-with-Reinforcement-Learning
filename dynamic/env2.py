# import numpy as np
# import warnings
# import matplotlib.pyplot as plt
# import random

# # Environment constants
# GRID_SIZE = 11
# CELL_SIZE = 10
# X_MAX_COORD = (GRID_SIZE - 1) * CELL_SIZE
# Y_MAX_COORD = (GRID_SIZE - 1) * CELL_SIZE
# OBSTACLE_WIDTH_COORD = CELL_SIZE
# AGENT_SPEED = CELL_SIZE
# OBSTACLE_SPEED = CELL_SIZE / 2

# final_route = {}

# class DynamicEnvironment:
#     def __init__(self, initial_position, target_position):
#         self.grid_size = GRID_SIZE
#         self.cell_size = CELL_SIZE
#         self.x_max_coord = X_MAX_COORD
#         self.y_max_coord = Y_MAX_COORD
#         self.obstacle_width_coord = OBSTACLE_WIDTH_COORD

#         self.max_coord = np.array([self.x_max_coord, self.y_max_coord])  # <-- Add this line

#         self.vector_initial_state = np.asarray(initial_position, dtype=float)
#         self.vector_terminal_state = np.asarray(target_position, dtype=float)
#         self.vector_agent_state = np.copy(self.vector_initial_state)

#         self.vision_range = 2 * CELL_SIZE

#         self.static_obstacles = [
#             [4 * CELL_SIZE, 4 * CELL_SIZE],
#             [5 * CELL_SIZE, 4 * CELL_SIZE],
#             [4 * CELL_SIZE, 5 * CELL_SIZE],
#             [5 * CELL_SIZE, 5 * CELL_SIZE],
#         ]

#         self.initial_obstacles = [
#             [3 * CELL_SIZE, 5 * CELL_SIZE, 1, 'horizontal'],
#             [6 * CELL_SIZE, 1 * CELL_SIZE, 1, 'vertical'],
#             [8 * CELL_SIZE, 8 * CELL_SIZE, 1, 'random'],
#         ]
#         self.obstacles = [list(obs) for obs in self.initial_obstacles]

#         self.agent_state_grid = np.zeros((2, self.grid_size, self.grid_size))
#         self._update_grid_state()

#         self.is_terminal = False
#         self.done_type = 0
#         self.steps_counter = 0
#         self.max_episode_steps = 500

#         self.action_space = {
#             0: [AGENT_SPEED, 0], 1: [0, AGENT_SPEED], 2: [-AGENT_SPEED, 0], 3: [0, -AGENT_SPEED],
#             4: [-AGENT_SPEED, AGENT_SPEED], 5: [-AGENT_SPEED, -AGENT_SPEED],
#             6: [AGENT_SPEED, AGENT_SPEED], 7: [AGENT_SPEED, -AGENT_SPEED]
#         }
#         self.num_actions = len(self.action_space)

#         self.current_path = {}
#         self.final_path = {}
#         self.path_index = 0
#         self.first_success = True
#         self.shortest_steps = float('inf')
#         self.longest_steps = 0

#         warnings.simplefilter("ignore", UserWarning)

#     def _map_coords_to_grid(self, x_coord, y_coord):
#         grid_x = int(round(x_coord / self.cell_size))
#         grid_y = int(round(y_coord / self.cell_size))
#         return max(0, min(self.grid_size - 1, grid_y)), max(0, min(self.grid_size - 1, grid_x))

#     def _update_grid_state(self):
#         self.agent_state_grid.fill(0)
#         for x, y in self.static_obstacles:
#             gy, gx = self._map_coords_to_grid(x, y)
#             self.agent_state_grid[1, gy, gx] = 1
#         for x, y, _, _ in self.obstacles:
#             gy, gx = self._map_coords_to_grid(x, y)
#             self.agent_state_grid[1, gy, gx] = 1
#         gy, gx = self._map_coords_to_grid(*self.vector_agent_state)
#         self.agent_state_grid[0, gy, gx] = 1

#     def _update_obstacles(self):
#         for i in range(len(self.obstacles)):
#             x, y, direction, motion_type = self.obstacles[i]

#             if motion_type == 'horizontal':
#                 x += direction * OBSTACLE_SPEED
#                 if x < 0 or x > self.x_max_coord:
#                     direction *= -1
#                     x = np.clip(x, 0, self.x_max_coord)

#             elif motion_type == 'vertical':
#                 y += direction * OBSTACLE_SPEED
#                 if y < 0 or y > self.y_max_coord:
#                     direction *= -1
#                     y = np.clip(y, 0, self.y_max_coord)

#             elif motion_type == 'random':
#                 x += random.choice([-1, 0, 1]) * OBSTACLE_SPEED
#                 y += random.choice([-1, 0, 1]) * OBSTACLE_SPEED
#                 x = np.clip(x, 0, self.x_max_coord)
#                 y = np.clip(y, 0, self.y_max_coord)

#             self.obstacles[i] = [x, y, direction, motion_type]

#     def is_collision(self, pos):
#         for x, y in self.static_obstacles + [obs[:2] for obs in self.obstacles]:
#             if np.linalg.norm(pos - np.array([x, y])) < self.obstacle_width_coord / 2:
#                 return True
#         return False

#     def is_terminal_reached(self):
#         return np.linalg.norm(self.vector_agent_state - self.vector_terminal_state) < (AGENT_SPEED / 2)

#     def get_reward(self, collision, reached):
#         if collision:
#             return -100, 'obstacle'
#         elif reached:
#             return 100, 'goal'
#         else:
#             d_goal = np.linalg.norm(self.vector_agent_state - self.vector_terminal_state)
#             d_start = np.linalg.norm(self.vector_agent_state - self.vector_initial_state)
#             max_dist = np.linalg.norm([self.x_max_coord, self.y_max_coord])
#             reward = 0.5 * (d_start / max_dist) + 1.5 * (1 - d_goal / max_dist)
#             return reward, 'continue'

#     def reset(self):
#         self.vector_agent_state = np.copy(self.vector_initial_state)
#         self.obstacles = [list(obs) for obs in self.initial_obstacles]
#         self._update_grid_state()

#         self.is_terminal = False
#         self.done_type = 0
#         self.steps_counter = 0
#         self.current_path = {}
#         self.path_index = 0

#         return self.vector_agent_state

#     def step(self, action):
#         if self.is_terminal:
#             return self.vector_agent_state, 'goal', 0, True, None

#         self._update_obstacles()
#         move = self.action_space[action]
#         new_pos = self.vector_agent_state + np.asarray(move, dtype=float)
#         new_pos = np.clip(new_pos, [0, 0], [self.x_max_coord, self.y_max_coord])

#         collision = self.is_collision(new_pos)
#         if not collision:
#             self.vector_agent_state = new_pos

#         self.is_terminal = self.is_terminal_reached()
#         self._update_grid_state()
#         reward, flag = self.get_reward(collision, self.is_terminal)

#         done = False
#         if collision:
#             self.done_type = -1
#             done = True
#         elif self.is_terminal:
#             self.done_type = 1
#             done = True
#         elif self.steps_counter >= self.max_episode_steps - 1:
#             done = True
#             flag = 'max_steps'

#         if not collision:
#             self.current_path[self.path_index] = self.vector_agent_state.tolist()
#             self.path_index += 1

#         if done and self.done_type == 1:
#             num_steps = len(self.current_path)
#             global final_route
#             if self.first_success:
#                 self.final_path = self.current_path.copy()
#                 self.shortest_steps = num_steps
#                 self.longest_steps = num_steps
#                 self.first_success = False
#             else:
#                 if num_steps < self.shortest_steps:
#                     self.shortest_steps = num_steps
#                     self.final_path = self.current_path.copy()
#                 if num_steps > self.longest_steps:
#                     self.longest_steps = num_steps
#             final_route = self.final_path

#         self.steps_counter += 1
#         return self.vector_agent_state, flag, reward, done, None

#     def render(self, episode_num=None, save_figure=True):
#         plt.figure(figsize=(8, 8))
#         ax = plt.gca()
#         ax.set_xlim(-CELL_SIZE/2, self.x_max_coord + CELL_SIZE/2)
#         ax.set_ylim(-CELL_SIZE/2, self.y_max_coord + CELL_SIZE/2)
#         ax.set_xticks(np.arange(0, self.x_max_coord + CELL_SIZE, self.cell_size))
#         ax.set_yticks(np.arange(0, self.y_max_coord + CELL_SIZE, self.cell_size))
#         plt.grid(True)

#         for x, y in self.static_obstacles:
#             ax.add_patch(plt.Rectangle((x - CELL_SIZE/2, y - CELL_SIZE/2), CELL_SIZE, CELL_SIZE, color='black'))

#         for x, y, *_ in self.obstacles:
#             ax.add_patch(plt.Rectangle((x - CELL_SIZE/2, y - CELL_SIZE/2), CELL_SIZE, CELL_SIZE, color='gray'))

#         sx, sy = self.vector_initial_state
#         tx, ty = self.vector_terminal_state
#         ax.add_patch(plt.Rectangle((sx - CELL_SIZE/2, sy - CELL_SIZE/2), CELL_SIZE, CELL_SIZE, color='green', label='Start'))
#         ax.add_patch(plt.Rectangle((tx - CELL_SIZE/2, ty - CELL_SIZE/2), CELL_SIZE, CELL_SIZE, color='red', label='Goal'))

#         ax.add_patch(plt.Rectangle((self.vector_agent_state[0] - CELL_SIZE/2, self.vector_agent_state[1] - CELL_SIZE/2),
#                                    CELL_SIZE, CELL_SIZE, color='blue', label='Agent'))

#         if self.final_path:
#             x_vals = [pos[0] for pos in self.final_path.values()]
#             y_vals = [pos[1] for pos in self.final_path.values()]
#             plt.plot(x_vals, y_vals, '-o', color='cyan', markersize=4)

#         plt.title(f"Dynamic Environment - Episode {episode_num}" if episode_num else "Dynamic Environment")
#         plt.legend()
#         if save_figure:
#             filename = f"dynamic_env_ep_{episode_num}.png" if episode_num else "dynamic_env.png"
#             plt.savefig(filename)
#             print(f"Saved: {filename}")
#         plt.show()

#     def simulate_obstacle_future(self, steps_ahead=2):
#         future_obstacles = [list(obs) for obs in self.obstacles]
#         simulated = []

#         for _ in range(steps_ahead):
#             temp = []
#             for i in range(len(future_obstacles)):
#                 x, y, direction, motion_type = future_obstacles[i]
#                 if motion_type == 'horizontal':
#                     x += direction * OBSTACLE_SPEED
#                     if x < 0 or x > self.x_max_coord:
#                         direction *= -1
#                         x = max(0, min(x, self.x_max_coord))

#                 elif motion_type == 'vertical':
#                     y += direction * OBSTACLE_SPEED
#                     if y < 0 or y > self.y_max_coord:
#                         direction *= -1
#                         y = max(0, min(y, self.y_max_coord))

#                 elif motion_type == 'random':
#                     x += random.choice([-1, 0, 1]) * OBSTACLE_SPEED
#                     y += random.choice([-1, 0, 1]) * OBSTACLE_SPEED
#                     x = max(0, min(x, self.x_max_coord))
#                     y = max(0, min(y, self.y_max_coord))

#                 temp.append([x, y, direction, motion_type])
#             simulated.append(temp)
#             future_obstacles = temp

#         return simulated

# def final_states():
#     return final_route


import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim

class DynamicEnvironment:
    def __init__(self, start, goal, grid_size=(100, 100), cell_size=10, vision_range=20):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.grid_size = np.array(grid_size)
        self.cell_size = cell_size
        self.vision_range = vision_range
        self.max_coord = self.grid_size[0]
        self.x_max_coord, self.y_max_coord = grid_size
        self.num_actions = 4
        self.action_space = {
            0: np.array([0, cell_size]),    # Up
            1: np.array([0, -cell_size]),   # Down
            2: np.array([-cell_size, 0]),   # Left
            3: np.array([cell_size, 0])     # Right
        }
        self.max_episode_steps = 50
        self.vector_initial_state = self.start.copy()
        self.vector_terminal_state = self.goal.copy()
        self.obstacles = []
        self.static_obstacles = [(30, 30), (40, 40), (50, 50)]
        self.reset()

    def reset(self):
        self.vector_initial_state = self.start.copy()
        self.vector_terminal_state = self.goal.copy()
        self.agent_pos = self.vector_initial_state.copy()
        self.steps = 0
        self.generate_dynamic_obstacles()
        return self.agent_pos

    def generate_dynamic_obstacles(self):
        self.obstacles = [(random.randint(0, self.x_max_coord // 10) * 10,
                           random.randint(0, self.y_max_coord // 10) * 10) for _ in range(5)]

    def step(self, action):
        move = self.action_space[action]
        next_pos = np.clip(self.agent_pos + move, 0, self.max_coord)
        self.agent_pos = next_pos.copy()
        self.steps += 1

        # Reward shaping based on proximity to goal and distance from start
        dist_to_goal = np.linalg.norm(self.agent_pos - self.vector_terminal_state)
        dist_from_start = np.linalg.norm(self.agent_pos - self.vector_initial_state)
        shaped_reward = -dist_to_goal / self.max_coord + 0.01 * dist_from_start

        done = False
        success = False
        collision = False

        if tuple(self.agent_pos) in self.obstacles or tuple(self.agent_pos) in self.static_obstacles:
            shaped_reward = -1.0
            collision = True
            done = True
        elif np.array_equal(self.agent_pos, self.vector_terminal_state):
            shaped_reward = 1.0
            success = True
            done = True
        elif self.steps >= self.max_episode_steps:
            done = True

        self.generate_dynamic_obstacles()  # Move obstacles
        return self.agent_pos, collision, shaped_reward, done, success

    def render(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.grid(True)

        # Obstacles
        for x, y in self.static_obstacles:
            ax.add_patch(plt.Rectangle((x - 5, y - 5), 10, 10, color='black'))
        for x, y in self.obstacles:
            ax.add_patch(plt.Rectangle((x - 5, y - 5), 10, 10, color='gray'))

        # Start and goal
        sx, sy = self.vector_initial_state
        gx, gy = self.vector_terminal_state
        ax.add_patch(plt.Rectangle((sx - 5, sy - 5), 10, 10, color='green'))
        ax.add_patch(plt.Rectangle((gx - 5, gy - 5), 10, 10, color='red'))

        # Agent
        ax.add_patch(plt.Rectangle((self.agent_pos[0] - 5, self.agent_pos[1] - 5), 10, 10, color='blue'))
        plt.title("Dynamic Environment")
        plt.show()

    def visualize(self, paths):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.grid(True)

        for x, y in self.static_obstacles:
            ax.add_patch(plt.Rectangle((x - 5, y - 5), 10, 10, color='black'))
        for x, y in self.obstacles:
            ax.add_patch(plt.Rectangle((x - 5, y - 5), 10, 10, color='gray'))

        sx, sy = self.vector_initial_state
        gx, gy = self.vector_terminal_state
        ax.add_patch(plt.Rectangle((sx - 5, sy - 5), 10, 10, color='green'))
        ax.add_patch(plt.Rectangle((gx - 5, gy - 5), 10, 10, color='red'))

        colors = {'Q-Learning': 'blue', 'SARSA': 'orange', 'DQN': 'purple'}
        for label, path in paths.items():
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, label=label, color=colors.get(label, 'black'))

        plt.legend()
        plt.title("Agent Paths")
        plt.show()
