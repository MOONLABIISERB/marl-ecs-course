import numpy as np
import matplotlib.pyplot as plt

# Grid and environment setup
grid_size = 9
start_state = (0, 0)  # Robot's starting position
goal_state = (8, 8)   # Goal position
obstacles = [(1, 3), (2, 3), (3, 3), (3, 2), (3, 1),
             (5, 5), (6, 5), (7, 5), (8, 5), (5, 6), (5, 7), (5, 8)]  # Obstacles
in_portal = (2, 2)  # Position of the IN portal
out_portal = (6, 6)  # Position of the OUT portal
actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # Actions: Down, Up, Left, Right
gamma = 0.9  # Discount factor for future rewards

# Function to check if a state is valid (within grid and not an obstacle)
def is_valid_state(s):
    return 0 <= s[0] < grid_size and 0 <= s[1] < grid_size and s not in obstacles

# Function to get the next state based on current state and action
def get_next_state(s, a):
    next_state = (s[0] + a[0], s[1] + a[1])
    # If the next state is invalid (out of bounds or an obstacle), stay in the same state
    if not is_valid_state(next_state):
        return s
    # Handle portal transitions
    if next_state == in_portal:
        return out_portal  # Entering the IN portal moves the agent to the OUT portal
    return next_state

# Value Iteration algorithm
def value_iteration():
    V = np.zeros((grid_size, grid_size))  # Initialize value function to zero
    policy = np.zeros((grid_size, grid_size, 2), dtype=int)  # Initialize policy
    
    while True:
        delta = 0  # Variable to track the maximum change in the value function
        for i in range(grid_size):
            for j in range(grid_size):
                s = (i, j)
                if s == goal_state or s in obstacles:  # Skip goal state and obstacles
                    continue
                v = V[i, j]
                Q = np.zeros(len(actions))  # Array to store Q-values for each action
                for a_idx, a in enumerate(actions):
                    next_state = get_next_state(s, a)
                    reward = 1 if next_state == goal_state else 0  # Reward is 1 only for the goal state
                    Q[a_idx] = reward + gamma * V[next_state[0], next_state[1]]  # Bellman equation
                V[i, j] = np.max(Q)  # Update the value with the maximum Q-value
                policy[i, j] = actions[np.argmax(Q)]  # Update policy with the best action
                delta = max(delta, abs(v - V[i, j]))  # Update delta with the largest value change
        if delta < 1e-4:  # Convergence check
            break
    
    return policy, V  # Return the optimal policy and value function

# Policy Iteration algorithm
def policy_iteration():
    # Initialize the policy with random actions
    policy = np.zeros((grid_size, grid_size, 2), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) != goal_state and (i, j) not in obstacles:
                policy[i, j] = actions[np.random.choice(len(actions))]

    V = np.zeros((grid_size, grid_size))  # Initialize value function to zero

    while True:
        # Policy Evaluation
        while True:
            delta = 0  # Variable to track the maximum change in the value function
            for i in range(grid_size):
                for j in range(grid_size):
                    s = (i, j)
                    if s == goal_state or s in obstacles:  # Skip goal state and obstacles
                        continue
                    a = tuple(policy[i, j])
                    next_state = get_next_state(s, a)
                    reward = 1 if next_state == goal_state else 0  # Reward is 1 only for the goal state
                    v = V[i, j]
                    V[i, j] = reward + gamma * V[next_state[0], next_state[1]]  # Bellman equation
                    delta = max(delta, abs(v - V[i, j]))  # Update delta with the largest value change
            if delta < 1e-4:  # Convergence check for policy evaluation
                break

        # Policy Improvement
        policy_stable = True  # Variable to check if policy is stable
        for i in range(grid_size):
            for j in range(grid_size):
                s = (i, j)
                if s == goal_state or s in obstacles:  # Skip goal state and obstacles
                    continue
                old_action = tuple(policy[i, j])
                Q = np.zeros(len(actions))  # Array to store Q-values for each action
                for a_idx, a in enumerate(actions):
                    next_state = get_next_state(s, a)
                    reward = 1 if next_state == goal_state else 0  # Reward is 1 only for the goal state
                    Q[a_idx] = reward + gamma * V[next_state[0], next_state[1]]  # Bellman equation
                new_action = actions[np.argmax(Q)]  # Get the best action from Q-values
                policy[i, j] = new_action  # Update the policy
                if old_action != new_action:  # If the policy changes, mark as unstable
                    policy_stable = False
        if policy_stable:  # If the policy is stable, stop the iteration
            break
    
    return policy, V  # Return the optimal policy and value function

# Function to visualize the policy using a quiver plot
def plot_policy(policy, title):
    plt.figure(figsize=(8, 8))
    grid_size = policy.shape[0]

    # Create a grid for quiver plot
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    U = np.zeros_like(X, dtype=float)  # X-direction vectors
    V = np.zeros_like(Y, dtype=float)  # Y-direction vectors

    # Fill in direction vectors based on policy
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == goal_state or (i, j) in obstacles or (i, j) == in_portal:
                continue  # Skip plotting arrows for goal, obstacles, and IN portal
            action = policy[i, j]
            U[i, j] = action[1]  # X component (horizontal direction)
            V[i, j] = action[0]  # Y component (vertical direction)

    # Plot the quiver arrows
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='blue')

    # Plot grid boundaries as full cells
    for x in range(grid_size + 1):
        plt.axhline(x - 0.5, color='black', linewidth=1)
    for y in range(grid_size + 1):
        plt.axvline(y - 0.5, color='black', linewidth=1)

    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.grid(False)  # Disable default grid
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')

    # Add goal marker
    plt.plot(goal_state[1], goal_state[0], 'r*', markersize=15)  # Goal with star symbol
    plt.text(goal_state[1], goal_state[0], 'Goal', fontsize=12, ha='center', va='center')

    # Add start marker
    plt.plot(start_state[1], start_state[0], 'go', markersize=10)  # Start with green circle
    plt.text(start_state[1], start_state[0], 'Start', fontsize=12, ha='center', va='center')

    # Add portals
    plt.plot(in_portal[1], in_portal[0], 'bs', markersize=10)  # IN portal with blue square
    plt.text(in_portal[1], in_portal[0], 'IN', fontsize=10, ha='center', va='center', color='white')
    plt.plot(out_portal[1], out_portal[0], 'bs', markersize=10)  # OUT portal with blue square
    plt.text(out_portal[1], out_portal[0], 'OUT', fontsize=10, ha='center', va='center', color='white')

    plt.show()

# Execute Value Iteration and plot the resulting policy
value_policy, _ = value_iteration()
plot_policy(value_policy, "Optimal Policy from Value Iteration")

# Execute Policy Iteration and plot the resulting policy
policy_policy, _ = policy_iteration()
plot_policy(policy_policy, "Optimal Policy from Policy Iteration")
