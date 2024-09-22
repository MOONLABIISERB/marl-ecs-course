import numpy as np
import matplotlib.pyplot as plt

# Define the grid-world environment
grid_size = 9
goal_state = (0, 8)
start_state = (8, 0)
tunnel_out = (2, 6)
tunnel_in = (6, 2)

# Define the actions (up, down, left, right)
actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

# Define the transition probabilities and rewards
transition_prob = 0.8
discount_factor = 0.9

# Initialize the value function and policy
V = np.zeros((grid_size, grid_size))
policy = np.zeros((grid_size, grid_size, 2))

# Perform Value Iteration
while True:
    V_new = np.copy(V)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == goal_state:
                continue
            max_value = float('-inf')
            for action_id, action in enumerate(actions):
                next_i = i + action[0]
                next_j = j + action[1]
                if 0 <= next_i < grid_size and 0 <= next_j < grid_size:
                    value = transition_prob * (0 + discount_factor * V[next_i, next_j])  # Reward is 0 except at goal
                    if value > max_value:
                        max_value = value
                        policy[i, j] = action
            V_new[i, j] = max_value
    if np.sum(np.abs(V_new - V)) < 1e-6:
        break
    V = np.copy(V_new)

# Plot the quiver plot to visualize the optimal policy
X, Y = np.meshgrid(np.arange(0, grid_size, 1), np.arange(0, grid_size, 1))
U = policy[:,:,0]
V = policy[:,:,1]

fig, ax = plt.subplots()
ax.quiver(X, Y, U, V)
ax.scatter(start_state[1], start_state[0], color='r', label='Start')
ax.scatter(goal_state[1], goal_state[0], color='g', label='Goal')
ax.scatter(tunnel_in[1], tunnel_in[0], color='b', label='Tunnel IN')
ax.scatter(tunnel_out[1], tunnel_out[0], color='orange', label='Tunnel OUT')
ax.legend()
plt.gca().invert_yaxis()  # Invert y-axis to match the grid layout
plt.show()
