import numpy as np
import matplotlib.pyplot as plt

# Define grid world
GRID_SIZE = 9
START = (0, 0)
GOAL = (8, 8)
TUNNEL_IN = (2, 2)
TUNNEL_OUT = (6, 6)
WALLS = [(3, 1), (3, 2), (3, 3), (1, 3), (2, 3),
         (5, 5), (5, 6), (5, 7), (5, 8), (6, 5),
         (7, 5), (8, 5)]

# Define actions
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
ACTION_PROB = 0.25  # Equal probability for each action
gamma = 0.9

# Initialize value function and policy
def initialize_grid():
    return np.zeros((GRID_SIZE, GRID_SIZE)), np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

def is_valid(state):
    return 0 <= state[0] < GRID_SIZE and 0 <= state[1] < GRID_SIZE and state not in WALLS

def get_next_state(state, action):
    if state == TUNNEL_IN:
        return TUNNEL_OUT
    next_state = (state[0] + action[0], state[1] + action[1])
    return next_state if is_valid(next_state) else state

def get_reward(state):
    return 1 if state == GOAL else 0

def value_iteration(gamma=0.9, epsilon=1e-6):
    V, policy = initialize_grid()
    while True:
        delta = 0
        new_V = np.copy(V)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) in WALLS or (i, j) == GOAL:
                    continue
                
                action_values = np.zeros(len(ACTIONS))
                for action_index, action in enumerate(ACTIONS):
                    next_state = get_next_state((i, j), action)
                    action_value = get_reward(next_state) + gamma * V[next_state]
                    action_values[action_index] = action_value
                
                max_value = np.max(action_values)
                new_V[i, j] = max_value
                delta = max(delta, abs(V[i, j] - new_V[i, j]))
        
        V = new_V
        if delta < epsilon:
            break
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) in WALLS or (i, j) == GOAL:
                continue
            
            action_values = np.zeros(len(ACTIONS))
            for action_index, action in enumerate(ACTIONS):
                next_state = get_next_state((i, j), action)
                action_value = get_reward(next_state) + gamma * V[next_state]
                action_values[action_index] = action_value
            
            max_value = np.max(action_values)
            best_actions = [index for index, value in enumerate(action_values) if value == max_value]
            policy[i, j] = best_actions[0]  # Choose the first optimal action

    return V, policy

def policy_iteration(gamma=0.9):
    V, policy = initialize_grid()
    while True:
        # Policy Evaluation
        while True:
            delta = 0
            new_V = np.copy(V)
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    if (i, j) in WALLS or (i, j) == GOAL:
                        continue
                    action = ACTIONS[policy[i, j]]
                    next_state = get_next_state((i, j), action)
                    new_V[i, j] = get_reward(next_state) + gamma * V[next_state]
                    delta = max(delta, abs(V[i, j] - new_V[i, j]))
            V = new_V
            if delta < 1e-6:
                break
        
        # Policy Improvement
        policy_stable = True
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) in WALLS or (i, j) == GOAL:
                    continue
                
                old_action = policy[i, j]
                action_values = np.zeros(len(ACTIONS))
                for action_index, action in enumerate(ACTIONS):
                    next_state = get_next_state((i, j), action)
                    action_value = get_reward(next_state) + gamma * V[next_state]
                    action_values[action_index] = action_value
                
                max_value = np.max(action_values)
                best_actions = [index for index, value in enumerate(action_values) if value == max_value]
                policy[i, j] = best_actions[0]  # Choose the first optimal action
                
                if old_action != policy[i, j]:
                    policy_stable = False

        if policy_stable:
            break

    return V, policy

def print_final_values(V, policy):
    print("Final Value Function:")
    for i in range(GRID_SIZE):
        row = [f'{V[i, j]:.2f}' if (i, j) not in WALLS and (i, j) != GOAL else '-----' for j in range(GRID_SIZE)]
        print(' '.join(row))
    
    print("\nFinal Policy:")
    action_symbols = ['→', '↓', '←', '↑']  # right, down, left, up
    for i in range(GRID_SIZE):
        row = [action_symbols[policy[i, j]] if (i, j) not in WALLS and (i, j) != GOAL else '-----' for j in range(GRID_SIZE)]
        print(' '.join(row))

def plot_policy(policy, V, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(np.arange(GRID_SIZE))
    ax.set_yticks(np.arange(GRID_SIZE))
    ax.grid(True)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) in WALLS:
                ax.add_patch(plt.Rectangle((j, GRID_SIZE-1-i), 1, 1, fill=True, color='orange'))
            elif (i, j) == GOAL:
                ax.text(j+0.5, GRID_SIZE-1-i+0.5, '★', ha='center', va='center', fontsize=20, color='purple')
            elif (i, j) == START:
                ax.text(j+0.5, GRID_SIZE-1-i+0.5, 'robot', ha='center', va='center', fontsize=20)
            elif (i, j) == TUNNEL_IN:
                ax.text(j+0.5, GRID_SIZE-1-i+0.5, 'IN', ha='center', va='center', fontsize=10, color='red')
            elif (i, j) == TUNNEL_OUT:
                ax.text(j+0.5, GRID_SIZE-1-i+0.5, 'OUT', ha='center', va='center', fontsize=10, color='blue')

            if (i, j) not in WALLS and (i, j) != GOAL:
                # Find all best actions
                action_values = np.zeros(len(ACTIONS))
                for action_index, action in enumerate(ACTIONS):
                    next_state = get_next_state((i, j), action)
                    action_value = get_reward(next_state) + gamma * V[next_state]
                    action_values[action_index] = action_value
                
                max_value = np.max(action_values)
                best_actions = [index for index, value in enumerate(action_values) if value == max_value]

                if (i, j) == TUNNEL_IN:
                    # Skip drawing arrows for TUNNEL_IN
                    continue
                
                for action_index in best_actions:
                    action = ACTIONS[action_index]
                    ax.quiver(j + 0.5, GRID_SIZE-1-i + 0.5, action[1], -action[0], 
                              angles='xy', scale_units='xy', scale=2, color='r')
                
                # Print value function at each cell
                ax.text(j + 0.5, GRID_SIZE-1-i + 0.5, f'{V[i, j]:.2f}', ha='center', va='center', fontsize=10, color='black')

    plt.gca().invert_yaxis()
    plt.show()

# Run Value Iteration
V_value_iter, policy_value_iter = value_iteration()
plot_policy(policy_value_iter, V_value_iter, "Optimal Policy (Value Iteration)")

# Run Policy Iteration
V_policy_iter, policy_policy_iter = policy_iteration()
plot_policy(policy_policy_iter, V_policy_iter, "Optimal Policy (Policy Iteration)")
