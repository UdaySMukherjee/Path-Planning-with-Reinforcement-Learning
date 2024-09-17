import numpy as np
import heapq
import time
from env import generate_env, visualize_env
import matplotlib.pyplot as plt
import random

# Define 4 possible movements: right, down, left, up
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    """Manhattan distance heuristic for 4-direction movement"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(env, start, end):
    """Perform A* algorithm for pathfinding with 4 directions"""
    start_time = time.time()
    start_node = Node(start)
    end_node = Node(end)
    
    open_list = []
    closed_list = set()

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        if current_node.position == end_node.position:
            return reconstruct_path(current_node), time.time() - start_time

        for direction in DIRECTIONS:
            neighbor_pos = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])

            if (0 <= neighbor_pos[0] < env.shape[0] and
                0 <= neighbor_pos[1] < env.shape[1] and
                env[neighbor_pos] != 1 and neighbor_pos not in closed_list):

                movement_cost = 1  # For 4-direction movement, cost is always 1
                neighbor_node = Node(neighbor_pos, current_node)
                neighbor_node.g = current_node.g + movement_cost
                neighbor_node.h = heuristic(neighbor_pos, end_node.position)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                heapq.heappush(open_list, neighbor_node)

    return None, time.time() - start_time

def reconstruct_path(current_node):
    """Reconstruct the path from start to end"""
    path = []
    while current_node:
        path.append(current_node.position)
        current_node = current_node.parent
    return path[::-1]

def visualize_path(env, path, filename="path_astar.png"):
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
    plt.title("Path Found by A*")
    
    # Save the plot as an image
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    size = random.choice([10, 15, 20, 25, 30])
    env = generate_env(size)
    visualize_env(env)  # Show the environment

    start = (0, 0)
    end = (size - 1, size - 1)

    # Run A* algorithm with 4-direction movement
    path, exec_time = astar(env, start, end)
    
    if path:
        path_length = len(path)
        print(f"Path found! Length: {path_length} steps")
        print(f"Execution time: {exec_time:.4f} seconds")

        # Visualize the best path found and save the path plot as an image
        visualize_path(env, path, filename="path_astar.png")
    else:
        print("No Path Found")

