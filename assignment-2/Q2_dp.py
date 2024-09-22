import numpy as np

grid = np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1],            
            [1, 0, 0, 1, 1, 1],
            [1, 3, 0, 0, 0, 1],
            [1, 0, 0, 2, 0, 1],
            [1, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ])

start_pos = (2, 3)  # Starting position of the agent
goal_pos = [(4, 2)]  # Goal position for the box

# Action space: UP, DOWN, LEFT, RIGHT
actions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}

# Check if the action is valid
def is_valid_move(pos, action):
    new_pos = (pos[0] + action[0], pos[1] + action[1])
    if 0 <= new_pos[0] < grid.shape[0] and 0 <= new_pos[1] < grid.shape[1]:
        return grid[new_pos] != -1
    return False


# Initialize value function and policy
V = np.zeros(grid.shape)
gamma = 0.9
theta = 1e-6

# Reward function
def reward(state):
    if state in goal_pos:
        return 0
    return -1

# Value Iteration
def value_iteration_sokoban(grid, gamma, theta):
    V = np.zeros(grid.shape)
    while True:
        delta = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = V[i][j]
                if grid[i][j] == -1:  # Wall
                    continue
                new_values = []
                for action in actions.values():
                    if is_valid_move((i, j), action):
                        new_pos = (i + action[0], j + action[1])
                        new_values.append(reward(new_pos) + gamma * V[new_pos[0]][new_pos[1]])
                V[i][j] = max(new_values) if new_values else v
                delta = max(delta, abs(v - V[i][j]))
        if delta < theta:
            break
    return V

optimal_values_sokoban = value_iteration_sokoban(grid, gamma, theta)
print("Dynamic Programming Optimal Value Function for Sokoban:", optimal_values_sokoban)
