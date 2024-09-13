import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time  # To measure execution time
from env import generate_env

# Define 8 possible movements: right, down, left, up, and diagonals
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

class Node:
    """A node class for Dijkstra Pathfinding"""
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Distance from start node

    def __lt__(self, other):
        return self.g < other.g

def dijkstra(env, start, end):
    """Perform Dijkstra's algorithm with 8 directions"""
    start_time = time.time()  # Start timing

    start_node = Node(start)
    end_node = Node(end)

    open_list = []
    closed_list = set()

    # Add the start node to the open list
    heapq.heappush(open_list, start_node)

    total_explored = 0  # Track the number of explored nodes

    while open_list:
        # Get the node with the lowest g score
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        total_explored += 1  # Increment the explored node count

        # Check if we have reached the end
        if current_node.position == end_node.position:
            end_time = time.time()  # End timing
            return reconstruct_path(current_node), total_explored, current_node.g, end_time - start_time

        # Explore neighbors in all 8 directions
        for direction in DIRECTIONS:
            neighbor_pos = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])

            # Ensure neighbor is within the bounds and is not an obstacle
            if (0 <= neighbor_pos[0] < env.shape[0] and
                0 <= neighbor_pos[1] < env.shape[1] and
                env[neighbor_pos] != 1 and
                neighbor_pos not in closed_list):
                
                # Set movement cost: 1 for straight, sqrt(2) for diagonal
                movement_cost = np.sqrt(2) if direction in [(1, 1), (-1, -1), (1, -1), (-1, 1)] else 1

                neighbor_node = Node(neighbor_pos, current_node)
                neighbor_node.g = current_node.g + movement_cost

                # Check if neighbor is already in open list with a better g score
                if any(neighbor.position == neighbor_pos and neighbor.g <= neighbor_node.g for neighbor in open_list):
                    continue
                
                heapq.heappush(open_list, neighbor_node)

    return None, total_explored, None, time.time() - start_time  # No path found

def reconstruct_path(current_node):
    """Reconstruct the path from start to end"""
    path = []
    while current_node:
        path.append(current_node.position)
        current_node = current_node.parent
    return path[::-1]  # Return reversed path

def visualize_path(env, path):
    """Visualize the environment and the path using arrows and lines"""
    cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()

    # Plot the grid environment
    ax.imshow(env, cmap=cmap, norm=norm, origin='upper')

    # Mark the start and end points
    ax.text(0, 0, 'Start', color='green', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(env.shape[1] - 1, env.shape[0] - 1, 'End', color='red', ha='center', va='center', fontsize=12, fontweight='bold')

    # Plot the path with arrows
    for i in range(len(path) - 1):
        start_pos = path[i]
        end_pos = path[i + 1]

        # Use arrows to indicate direction
        ax.annotate(
            '', xy=(end_pos[1], end_pos[0]), xytext=(start_pos[1], start_pos[0]),
            arrowprops=dict(facecolor='blue', edgecolor='blue', shrink=0.05, width=1.5, headwidth=7)
        )

    # Set grid for better visualization
    ax.grid(True, which='both', color='gray', linewidth=0.5, linestyle='--')
    ax.set_xticks(np.arange(-0.5, env.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, env.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title("Dijkstra Pathfinding with Arrows")
    plt.show()

if __name__ == "__main__":
    # Generate environment from env.py
    env = generate_env()

    # Define start and end points
    start = (0, 0)
    end = (env.shape[0] - 1, env.shape[1] - 1)

    # Find the shortest path using Dijkstra
    path, explored_nodes, total_cost, exec_time = dijkstra(env, start, end)

    if path:
        print("Path found:", path)
        print(f"Path length (steps): {len(path)}")
        print(f"Total path cost: {total_cost}")
        print(f"Number of explored nodes: {explored_nodes}")
        print(f"Execution time: {exec_time:.4f} seconds")
        # Visualize the path with arrows and lines
        visualize_path(env, path)
    else:
        print("No path found")
        print(f"Number of explored nodes: {explored_nodes}")
        print(f"Execution time: {exec_time:.4f} seconds")
