import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

# Generate environment function
def generate_env(size):
    """Generate an environment with random obstacles"""
    obstacle_prob = random.uniform(0.1, 0.4)
    env = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if np.random.rand() < obstacle_prob:
                env[i, j] = 1

    # Start and end points
    env[0, 0] = 2  # Start
    env[size-1, size-1] = 3  # End

    return env

# Visualize environment and save as env.png
def visualize_env(env, filename="env.png"):
    """Visualize the environment with obstacles, start, and end points"""
    cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(env, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0, 1, 2, 3], label='Grid Elements')
    plt.grid(True, which='both', color='gray', linestyle='--')
    plt.savefig(filename)
    plt.close()

# Generate and visualize the environment
size = random.choice([10, 15, 20, 25, 30])
env = generate_env(size)
visualize_env(env, "env.png")
