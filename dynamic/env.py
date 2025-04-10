import numpy as np
import warnings
import matplotlib.pyplot as plt
import random

# Environment constants
GRID_SIZE = 11
CELL_SIZE = 10
X_MAX_COORD = (GRID_SIZE - 1) * CELL_SIZE
Y_MAX_COORD = (GRID_SIZE - 1) * CELL_SIZE
OBSTACLE_WIDTH_COORD = CELL_SIZE
AGENT_SPEED = CELL_SIZE
OBSTACLE_SPEED = CELL_SIZE / 2

final_route = {}

class DynamicEnvironment:
    def __init__(self, initial_position, target_position):
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.x_max_coord = X_MAX_COORD
        self.y_max_coord = Y_MAX_COORD
        self.obstacle_width_coord = OBSTACLE_WIDTH_COORD

        self.vector_initial_state = np.asarray(initial_position, dtype=float)
        self.vector_terminal_state = np.asarray(target_position, dtype=float)
        self.vector_agent_state = np.copy(self.vector_initial_state)

        self.static_obstacles = [
            [4 * CELL_SIZE, 4 * CELL_SIZE],
            [5 * CELL_SIZE, 4 * CELL_SIZE],
            [4 * CELL_SIZE, 5 * CELL_SIZE],
            [5 * CELL_SIZE, 5 * CELL_SIZE],
        ]

        self.initial_obstacles = [
            [3 * CELL_SIZE, 5 * CELL_SIZE, 1, 'horizontal'],
            [6 * CELL_SIZE, 1 * CELL_SIZE, 1, 'vertical'],
            [8 * CELL_SIZE, 8 * CELL_SIZE, 1, 'random'],
        ]
        self.obstacles = [list(obs) for obs in self.initial_obstacles]

        self.agent_state_grid = np.zeros((2, self.grid_size, self.grid_size))
        self._update_grid_state()

        self.is_terminal = False
        self.done_type = 0
        self.steps_counter = 0
        self.max_episode_steps = 500

        self.action_space = {
            0: [AGENT_SPEED, 0], 1: [0, AGENT_SPEED], 2: [-AGENT_SPEED, 0], 3: [0, -AGENT_SPEED],
            4: [-AGENT_SPEED, AGENT_SPEED], 5: [-AGENT_SPEED, -AGENT_SPEED],
            6: [AGENT_SPEED, AGENT_SPEED], 7: [AGENT_SPEED, -AGENT_SPEED]
        }
        self.num_actions = len(self.action_space)

        self.current_path = {}
        self.final_path = {}
        self.path_index = 0
        self.first_success = True
        self.shortest_steps = float('inf')
        self.longest_steps = 0

        warnings.simplefilter("ignore", UserWarning)

    def _map_coords_to_grid(self, x_coord, y_coord):
        grid_x = int(round(x_coord / self.cell_size))
        grid_y = int(round(y_coord / self.cell_size))
        return max(0, min(self.grid_size - 1, grid_y)), max(0, min(self.grid_size - 1, grid_x))

    def _update_grid_state(self):
        self.agent_state_grid.fill(0)
        for x, y in self.static_obstacles:
            gy, gx = self._map_coords_to_grid(x, y)
            self.agent_state_grid[1, gy, gx] = 1
        for x, y, _, _ in self.obstacles:
            gy, gx = self._map_coords_to_grid(x, y)
            self.agent_state_grid[1, gy, gx] = 1
        gy, gx = self._map_coords_to_grid(*self.vector_agent_state)
        self.agent_state_grid[0, gy, gx] = 1

    def _update_obstacles(self):
        for i in range(len(self.obstacles)):
            x, y, direction, motion_type = self.obstacles[i]

            if motion_type == 'horizontal':
                x += direction * OBSTACLE_SPEED
                if x < 0 or x > self.x_max_coord:
                    direction *= -1
                    x = max(0, min(x, self.x_max_coord))

            elif motion_type == 'vertical':
                y += direction * OBSTACLE_SPEED
                if y < 0 or y > self.y_max_coord:
                    direction *= -1
                    y = max(0, min(y, self.y_max_coord))

            elif motion_type == 'random':
                x += random.choice([-1, 0, 1]) * OBSTACLE_SPEED
                y += random.choice([-1, 0, 1]) * OBSTACLE_SPEED
                x = max(0, min(x, self.x_max_coord))
                y = max(0, min(y, self.y_max_coord))

            self.obstacles[i] = [x, y, direction, motion_type]

    def is_collision(self, agent_pos_vector):
        agent_x, agent_y = agent_pos_vector
        agent_radius = AGENT_SPEED / 2
        obs_radius = self.obstacle_width_coord / 2

        for obs_x, obs_y in self.static_obstacles:
            if (agent_x - obs_x)**2 + (agent_y - obs_y)**2 < (agent_radius + obs_radius)**2:
                return True

        for obs_x, obs_y, *_ in self.obstacles:
            if (agent_x - obs_x)**2 + (agent_y - obs_y)**2 < (agent_radius + obs_radius)**2:
                return True
        return False

    def is_terminal_reached(self):
        return np.linalg.norm(self.vector_agent_state - self.vector_terminal_state) < (AGENT_SPEED / 2)

    def get_reward(self, collision, reached):
        if collision:
            return -100, 'obstacle'
        elif reached:
            return 100, 'goal'
        else:
            return -1, 'continue'

    def reset(self):
        self.vector_agent_state = np.copy(self.vector_initial_state)
        self.obstacles = [list(obs) for obs in self.initial_obstacles]
        self._update_grid_state()

        self.is_terminal = False
        self.done_type = 0
        self.steps_counter = 0
        self.current_path = {}
        self.path_index = 0

        return self.vector_agent_state

    def step(self, action):
        if self.is_terminal:
            return self.vector_agent_state, 'goal', 0, True, None

        self._update_obstacles()
        move = self.action_space[action]
        new_pos = self.vector_agent_state + np.asarray(move, dtype=float)
        new_pos[0] = np.clip(new_pos[0], 0, self.x_max_coord)
        new_pos[1] = np.clip(new_pos[1], 0, self.y_max_coord)

        collision = self.is_collision(new_pos)
        if not collision:
            self.vector_agent_state = new_pos

        self.is_terminal = self.is_terminal_reached()
        self._update_grid_state()
        reward, flag = self.get_reward(collision, self.is_terminal)

        done = False
        if collision:
            self.done_type = -1
            done = True
        elif self.is_terminal:
            self.done_type = 1
            done = True
        elif self.steps_counter >= self.max_episode_steps - 1:
            done = True
            flag = 'max_steps'

        if not collision:
            self.current_path[self.path_index] = self.vector_agent_state.tolist()
            self.path_index += 1

        if done and self.done_type == 1:
            num_steps = len(self.current_path)
            global final_route
            if self.first_success:
                self.final_path = self.current_path.copy()
                self.shortest_steps = num_steps
                self.longest_steps = num_steps
                self.first_success = False
            else:
                if num_steps < self.shortest_steps:
                    self.shortest_steps = num_steps
                    self.final_path = self.current_path.copy()
                if num_steps > self.longest_steps:
                    self.longest_steps = num_steps
            final_route = self.final_path

        self.steps_counter += 1
        return self.vector_agent_state, flag, reward, done, None

    def render(self, episode_num=None, save_figure=True):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_xlim(-CELL_SIZE/2, self.x_max_coord + CELL_SIZE/2)
        ax.set_ylim(-CELL_SIZE/2, self.y_max_coord + CELL_SIZE/2)
        ax.set_xticks(np.arange(0, self.x_max_coord + CELL_SIZE, self.cell_size))
        ax.set_yticks(np.arange(0, self.y_max_coord + CELL_SIZE, self.cell_size))
        plt.grid(True)

        for x, y in self.static_obstacles:
            ax.add_patch(plt.Rectangle((x - CELL_SIZE/2, y - CELL_SIZE/2), CELL_SIZE, CELL_SIZE, color='black', label='Static Obstacle' if 'Static Obstacle' not in ax.get_legend_handles_labels()[1] else ""))

        for x, y, *_ in self.obstacles:
            ax.add_patch(plt.Rectangle((x - CELL_SIZE/2, y - CELL_SIZE/2), CELL_SIZE, CELL_SIZE, color='gray', label='Dynamic Obstacle' if 'Dynamic Obstacle' not in ax.get_legend_handles_labels()[1] else ""))

        sx, sy = self.vector_initial_state
        plt.scatter(sx, sy, color='green', s=150, label='Start', zorder=5)
        tx, ty = self.vector_terminal_state
        plt.scatter(tx, ty, color='red', s=150, label='Terminal', zorder=5)
        ax.scatter(*self.vector_agent_state, color='blue', s=100, label='Agent', zorder=5)

        if self.final_path:
            x_vals = [pos[0] for pos in self.final_path.values()]
            y_vals = [pos[1] for pos in self.final_path.values()]
            plt.plot(x_vals, y_vals, '-o', color='cyan', markersize=4, label=f"Final Path ({self.shortest_steps} steps)", zorder=3)

        plt.title(f"City-Like Dynamic Environment - Episode {episode_num}" if episode_num else "City-Like Dynamic Environment")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        if save_figure:
            filename = f"city_env_ep_{episode_num}.png" if episode_num else "city_env.png"
            plt.savefig(filename)
            print(f"Saved: {filename}")
        plt.show()

def final_states():
    return final_route
