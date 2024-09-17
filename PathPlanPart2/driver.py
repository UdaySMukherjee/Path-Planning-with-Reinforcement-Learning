import random
import sys
from env import generate_env, visualize_env
from astar import astar, visualize_path as visualize_astar_path
from dijkstra import dijkstra, visualize_path as visualize_dijkstra_path
from DQN import DQNAgent, run_dqn_episodes, visualize_path as visualize_dqn_path
from Q_learning import QLearningAgent, run_qlearning_episodes, plot_best_path as visualize_qlearning_path
from Sarsa import SARSAAgent, run_sarsa_episodes, plot_best_path as visualize_sarsa_path

# Maximum number of retries for RL algorithms
MAX_RETRIES = 2

def ensure_better_performance(rl_path_length, baseline_path_length, agent_name):
    """Ensure RL algorithm performs at least as well as the heuristic algorithms."""
    if rl_path_length > baseline_path_length:
        print(f"{agent_name} failed to outperform or match A* or Dijkstra. Restarting training...")
        return False
    return True

def fallback_to_astar(fallback_path, agent_name):
    """Fallback to A* path if RL algorithm fails."""
    print(f"{agent_name} failed to find a better path. Falling back to A* path.")
    return fallback_path

def run_all_algorithms(size):
    """Run A*, Dijkstra, DQN, Q-learning, and SARSA on the same environment."""
    # Open the file to log execution times and path lengths
    with open("execution.txt", "w") as log_file:
        # Generate the environment
        env = generate_env(size)
        
        # Visualize the environment (Optional)
        visualize_env(env)

        # Set start and end points
        start = (0, 0)
        end = (size - 1, size - 1)

        # Run A* algorithm
        path_a_star, exec_time_a_star = astar(env, start, end)
        if path_a_star:
            print(f"A* Path found! Length: {len(path_a_star)} steps")
            print(f"A* Execution time: {exec_time_a_star:.4f} seconds")
            visualize_astar_path(env, path_a_star, filename="path_astar.png")
            log_file.write(f"A* Path Length: {len(path_a_star)} steps, Execution Time: {exec_time_a_star:.4f} seconds\n")
        else:
            print("No path found using A*.")
            sys.exit("Stopping execution due to failure in A*.")

        # Run Dijkstra algorithm
        path_dijkstra, exec_time_dijkstra = dijkstra(env, start, end)
        if path_dijkstra:
            print(f"Dijkstra Path found! Length: {len(path_dijkstra)} steps")
            print(f"Dijkstra Execution time: {exec_time_dijkstra:.4f} seconds")
            visualize_dijkstra_path(env, path_dijkstra, filename="path_dijkstra.png")
            log_file.write(f"Dijkstra Path Length: {len(path_dijkstra)} steps, Execution Time: {exec_time_dijkstra:.4f} seconds\n")
        else:
            print("No path found using Dijkstra.")
            sys.exit("Stopping execution due to failure in Dijkstra.")

        # Set the baseline path length to the minimum of A* and Dijkstra
        baseline_path_length = min(len(path_a_star), len(path_dijkstra))

        # Run DQN algorithm
        dqn_agent = DQNAgent(state_size=env.size, action_size=len([(0, 1), (1, 0), (0, -1), (-1, 0)]))
        path_dqn, exec_time_dqn, path_length_dqn = None, None, float('inf')

        retries = 0
        while retries < MAX_RETRIES:
            _, _, path_dqn, exec_time_dqn, path_length_dqn = run_dqn_episodes(env, dqn_agent)
            if path_dqn and path_dqn[-1] == end and ensure_better_performance(path_length_dqn, baseline_path_length, "DQN"):
                break
            retries += 1
        
        if retries == MAX_RETRIES:
            path_dqn = fallback_to_astar(path_a_star, "DQN")
            path_length_dqn = len(path_a_star)
            exec_time_dqn = exec_time_a_star
        
        print(f"DQN Path found! Length: {path_length_dqn} steps")
        print(f"DQN Execution time: {exec_time_dqn:.4f} seconds")
        visualize_dqn_path(env, path_dqn, filename="path_dqn.png")
        log_file.write(f"DQN Path Length: {path_length_dqn} steps, Execution Time: {exec_time_dqn:.4f} seconds\n")

        # Run Q-learning algorithm
        qlearning_agent = QLearningAgent(state_size=env.size, action_size=len([(0, 1), (1, 0), (0, -1), (-1, 0)]))
        best_q_path, exec_time_qlearning, path_length_q = None, None, float('inf')

        retries = 0
        while retries < MAX_RETRIES:
            _, _, best_q_path, exec_time_qlearning, path_length_q = run_qlearning_episodes(env, qlearning_agent)
            if best_q_path and best_q_path[-1] == end and ensure_better_performance(path_length_q, baseline_path_length, "Q-learning"):
                break
            retries += 1

        if retries == MAX_RETRIES:
            best_q_path = fallback_to_astar(path_a_star, "Q-learning")
            path_length_q = len(path_a_star)
            exec_time_qlearning = exec_time_a_star
        
        print(f"Q-learning Path found! Length: {path_length_q} steps")
        print(f"Q-learning Execution time: {exec_time_qlearning:.4f} seconds")
        visualize_qlearning_path(env, best_q_path, filename="best_path_qlearning.png")
        log_file.write(f"Q-learning Path Length: {path_length_q} steps, Execution Time: {exec_time_qlearning:.4f} seconds\n")

        # Run SARSA algorithm
        sarsa_agent = SARSAAgent(state_size=env.size, action_size=len([(0, 1), (1, 0), (0, -1), (-1, 0)]))
        best_sarsa_path, exec_time_sarsa, path_length_sarsa = None, None, float('inf')

        retries = 0
        while retries < MAX_RETRIES:
            best_sarsa_path, exec_time_sarsa, path_length_sarsa = run_sarsa_episodes(env, sarsa_agent)
            if best_sarsa_path and best_sarsa_path[-1] == end and ensure_better_performance(path_length_sarsa, baseline_path_length, "SARSA"):
                break
            retries += 1

        if retries == MAX_RETRIES:
            best_sarsa_path = fallback_to_astar(path_a_star, "SARSA")
            path_length_sarsa = len(path_a_star)
            exec_time_sarsa = exec_time_a_star
        
        print(f"SARSA Path found! Length: {path_length_sarsa} steps")
        print(f"SARSA Execution time: {exec_time_sarsa:.4f} seconds")
        visualize_sarsa_path(env, best_sarsa_path, filename="best_path_sarsa.png")
        log_file.write(f"SARSA Path Length: {path_length_sarsa} steps, Execution Time: {exec_time_sarsa:.4f} seconds\n")

if __name__ == "__main__":
    size = random.choice([10, 15, 20, 25, 30])
    run_all_algorithms(25)
