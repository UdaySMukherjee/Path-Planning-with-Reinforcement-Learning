import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_env(size=25, obstacle_prob=0.2):
    # Create a 25x25 matrix filled with zeros (0 means free space)
    env = np.zeros((size, size))

    # Place obstacles randomly (1 means obstacle)
    for i in range(size):
        for j in range(size):
            if np.random.rand() < obstacle_prob:
                env[i, j] = 1

    # Set start and end points
    env[0, 0] = 2  # Start point (represented by 2)
    env[size-1, size-1] = 3  # End point (represented by 3)

    return env

def visualize_env(env):
    # Define a custom colormap for the environment
    cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create the plot
    plt.imshow(env, cmap=cmap, norm=norm, origin='upper')

    # Create a color bar with labels
    cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5], label='Environment Elements')
    cbar.ax.set_yticklabels(['Free Space', 'Obstacle', 'Start', 'End'])

    # Add title and display the plot
    plt.title("Environment with Start, End, and Obstacles")
    plt.show()

if __name__ == "__main__":
    # Generate the environment
    env = generate_env()

    # Visualize the environment
    visualize_env(env)
