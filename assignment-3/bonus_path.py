from bonus_train import ACTIONS, maze_env, q_tables
import numpy as np
import matplotlib.pyplot as plt
from bonus_env import agent_colors, maze_size, maze, patches


def print_optimal_paths(maze_env, q_tables):
    """
    Print the optimal path taken by each agent based on their respective Q-table.
    """
    for i in range(maze_env.num_agents):
        print(f"\nAgent {i+1} (Color: {agent_colors[i]}):")
        current_pos = maze_env.agents[i]
        destination = maze_env.destinations[i]
        path = [current_pos]

        while current_pos != destination:
            # Get the best action from the Q-table
            best_action_index = np.argmax(q_tables[i][current_pos])
            move = ACTIONS[best_action_index]
            next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

            # Check if the move is valid
            if maze_env.is_valid_position(next_pos):
                current_pos = next_pos
                path.append(current_pos)
            else:
                print(f"Invalid move encountered at {current_pos}. Exiting.")
                break

        print(f"Optimal Path: {path}")

print_optimal_paths(maze_env,q_tables)


def visualize_optimal_paths_with_arrows(maze, maze_env, q_tables):
    """
    Visualize the maze with optimal paths for all agents, including arrows for direction.
    """
    # Create a copy of the maze for visualization
    path_grid = np.copy(maze)
    agent_colors = ['red', 'blue', 'green', 'purple']
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the maze grid
    for x in range(maze_size):
        for y in range(maze_size):
            if maze[x, y] == 1:  # Wall
                ax.add_patch(patches.Rectangle((y, maze_size - x - 1), 1, 1, color='brown'))
            elif maze[x, y] == 3:  # Destination
                agent_index = maze_env.destinations.index((x, y))
                color = agent_colors[agent_index]
                ax.plot(y + 0.5, maze_size - x - 1 + 0.5, marker='x', color=color, markersize=15, markeredgewidth=3)

    # Compute and visualize paths for all agents
    for i in range(maze_env.num_agents):
        current_pos = maze_env.agents[i]
        destination = maze_env.destinations[i]
        path_x, path_y = [], []

        # Collect path coordinates
        path_x.append(current_pos[1] + 0.5)
        path_y.append(maze_size - current_pos[0] - 1 + 0.5)

        while current_pos != destination:
            best_action_index = np.argmax(q_tables[i][current_pos])
            move = ACTIONS[best_action_index]
            next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

            if maze_env.is_valid_position(next_pos):
                # Add arrow for direction
                ax.arrow(
                    current_pos[1] + 0.5,
                    maze_size - current_pos[0] - 1 + 0.5,
                    move[1] * 0.8,  # Scaled to fit inside the grid
                    -move[0] * 0.8,  # Negative because of grid's flipped y-axis
                    head_width=0.2,
                    head_length=0.2,
                    fc=agent_colors[i],
                    ec=agent_colors[i],
                )
                current_pos = next_pos
                path_x.append(current_pos[1] + 0.5)
                path_y.append(maze_size - current_pos[0] - 1 + 0.5)
            else:
                print(f"Invalid move encountered for Agent {i+1} at {current_pos}.")
                break

        # Plot the path for the agent
        ax.plot(path_x, path_y, color=agent_colors[i], linestyle='-', linewidth=2, label=f'Agent {i+1}')
        ax.add_patch(patches.Circle((path_x[0], path_y[0]), 0.3, color=agent_colors[i]))  # Start
        ax.add_patch(patches.Circle((path_x[-1], path_y[-1]), 0.3, color=agent_colors[i], alpha=0.5))  # Goal

    # Add grid lines
    for i in range(maze_size + 1):
        ax.plot([0, maze_size], [i, i], color='black', linewidth=0.5)
        ax.plot([i, i], [0, maze_size], color='black', linewidth=0.5)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=8)
    plt.title("Optimal Paths with Direction for All Agents")
    plt.show()

# Visualize the optimal paths with arrows
visualize_optimal_paths_with_arrows(maze, maze_env, q_tables)
