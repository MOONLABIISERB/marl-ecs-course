import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import numpy as np

# Initialize the maze dimensions and empty grid
maze_size = 10
maze = np.zeros((maze_size, maze_size), dtype=int)

# Define custom walls (1 indicates a wall)
# Example: Add some walls manually
walls = [
    (0, 4), (1, 4), (2, 4), (2, 5),
    (4, 7), (4, 8), (4, 9), (5, 7),
    (4, 0), (4, 1), (4, 2), (5, 2),
    (7, 5), (8, 5), (9, 5), (7, 4)
]
for wall in walls:
    maze[wall] = 1

# Randomly initialize agents (2 indicates an agent)
num_agents = 4
agents = []
while len(agents) < num_agents:
    x, y = random.randint(0, maze_size-1), random.randint(0, maze_size-1)
    if maze[x, y] == 0:  # Ensure the position is free
        agents.append((x, y))
        maze[x, y] = 2

# Define custom destinations (3 indicates a destination)
destinations = [(1, 5), (5, 8), (5, 1), (8, 4)]
for dest in destinations:
    if maze[dest] == 0:  # Ensure the position is free
        maze[dest] = 3



# Print agent and destination positions
print("\nAgents' positions (row, column):", agents)
print("Destinations' positions (row, column):", destinations)

# Define colors for agents
agent_colors = ['red', 'blue', 'green', 'purple']

# Function to plot the maze
def plot_maze(maze, agents, destinations):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, maze_size)
    ax.set_ylim(0, maze_size)

    # Draw the grid
    for x in range(maze_size):
        for y in range(maze_size):
            if maze[x, y] == 1:  # Walls
                ax.add_patch(patches.Rectangle((y, maze_size-x-1), 1, 1, color='brown'))
            elif maze[x, y] == 2:  # Agents
                agent_index = agents.index((x, y))
                ax.add_patch(patches.Circle((y+0.5, maze_size-x-1+0.5), 0.3, color=agent_colors[agent_index]))
            elif maze[x, y] == 3:  # Destinations
                agent_index = destinations.index((x, y))
                color = agent_colors[agent_index]
                ax.plot(y+0.5, maze_size-x-1+0.5, marker='x', color=color, markersize=15, markeredgewidth=3)

    # Draw grid lines
    for i in range(maze_size+1):
        ax.plot([0, maze_size], [i, i], color='black', linewidth=0.5)
        ax.plot([i, i], [0, maze_size], color='black', linewidth=0.5)

    ax.set_aspect('equal')
    ax.axis('off')  # Turn off axes for a cleaner look
    plt.title("Maze Visualization")
    plt.show()

# Call the plotting function
plot_maze(maze, agents, destinations)