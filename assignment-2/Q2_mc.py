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


import random

def monte_carlo_sokoban(episodes, gamma):
    V_mc = np.zeros(grid.shape)
    returns = {state: [] for state in np.ndindex(grid.shape)}

    for episode in range(episodes):
        total_reward = 0
        states_visited = []
        pos = start_pos

        while True:
            action = random.choice(list(actions.values()))
            if is_valid_move(pos, action):
                new_pos = (pos[0] + action[0], pos[1] + action[1])
                reward_value = reward(new_pos)
                total_reward += reward_value
                states_visited.append(pos)
                pos = new_pos
                if pos in goal_pos:
                    break
        for state in states_visited:
            G = total_reward * (gamma ** states_visited.index(state))
            returns[state].append(G)
            V_mc[state] = np.mean(returns[state])

    return V_mc

optimal_mc_values_sokoban = monte_carlo_sokoban(episodes=1000, gamma=0.9)
print("Monte Carlo Optimal Values for Sokoban:", optimal_mc_values_sokoban)
