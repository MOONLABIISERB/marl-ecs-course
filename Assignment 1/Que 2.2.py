import numpy as np
import matplotlib.pyplot as plt

grid_size = 9
num_states = grid_size * grid_size

goal_state = 8
blocked_cells = [5, 14, 23, 32, 33, 34, 35, 46, 47, 48, 57, 66]
in_tunnel = 56
out_tunnel = 24
start_state = 72

rewards = np.zeros(num_states)
rewards[goal_state] = 1

actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
num_actions = len(actions)

def get_next_state(state, action):
    row = state // grid_size
    col = state % grid_size
    new_row = row + action[0]
    new_col = col + action[1]
    
    if new_row < 0 or new_row >= grid_size or new_col < 0 or new_col >= grid_size:
        return state  
    
    next_state = new_row * grid_size + new_col
    if next_state in blocked_cells:
        return state  
    
    if next_state == in_tunnel:
        return out_tunnel  
    
    return next_state

action_vectors = {
    0: (-1, 0),  
    1: (1, 0),   
    2: (0, -1),   
    3: (0, 1),    
    4: (-1, -1),  
    5: (-1, 1),   
    6: (1, -1),   
    7: (1, 1)    
}

def value_iteration(gamma=0.9, theta=1e-5):
    value_function = np.zeros(num_states)
    
    while True:
        delta = 0
        new_value_function = np.zeros(num_states)
        
        for state in range(num_states):
            if state in blocked_cells or state == goal_state:
                continue
            
            max_value = float('-inf')
            for action in actions:
                next_state = get_next_state(state, action)
                value = rewards[next_state] + gamma * value_function[next_state]
                if value > max_value:
                    max_value = value
            
            new_value_function[state] = max_value
            delta = max(delta, abs(value_function[state] - new_value_function[state]))
        
        value_function = new_value_function
        
        if delta < theta:
            break
    
    policy = np.zeros(num_states, dtype=int)
    for state in range(num_states):
        if state in blocked_cells or state == goal_state:
            continue
        
        max_value = float('-inf')
        best_action = 0
        for i, action in enumerate(actions):
            next_state = get_next_state(state, action)
            value = rewards[next_state] + gamma * value_function[next_state]
            if value > max_value:
                max_value = value
                best_action = i
        
        policy[state] = best_action
    
    return policy, value_function

def policy_iteration(gamma=0.9, theta=1e-5):
    policy = np.random.choice(num_actions, num_states)
    value_function = np.zeros(num_states)
    
    while True:
    
        while True:
            delta = 0
            for state in range(num_states):
                if state in blocked_cells or state == goal_state:
                    continue
                
                action = actions[policy[state]]
                next_state = get_next_state(state, action)
                value = rewards[next_state] + gamma * value_function[next_state]
                delta = max(delta, abs(value_function[state] - value))
                value_function[state] = value
            
            if delta < theta:
                break
        
        
        policy_stable = True
        for state in range(num_states):
            if state in blocked_cells or state == goal_state:
                continue
            
            old_action = policy[state]
            max_value = float('-inf')
            best_action = old_action
            for i, action in enumerate(actions):
                next_state = get_next_state(state, action)
                value = rewards[next_state] + gamma * value_function[next_state]
                if value > max_value:
                    max_value = value
                    best_action = i
            
            policy[state] = best_action
            if best_action != old_action:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy, value_function

value_policy, value_function = value_iteration()
policy_policy, policy_function = policy_iteration()

def visualize_policy(policy, title):
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    
    for state in range(num_states):
        if state in blocked_cells or state == goal_state:
            continue
        row = state // grid_size
        col = state % grid_size
        action = policy[state]
        direction = action_vectors[action]
        U[row, col] = direction[1]
        V[row, col] = -direction[0]  
    
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V, color='r')
    plt.scatter(goal_state % grid_size, goal_state // grid_size, color='g', marker='*', s=200, label='Goal')
    plt.scatter(start_state % grid_size, start_state // grid_size, color='b', marker='o', s=100, label='Start')
    for blocked in blocked_cells:
        plt.scatter(blocked % grid_size, blocked // grid_size, color='black', marker='x', s=100)
    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_policy(value_policy, "Optimal Policy from Value Iteration")
visualize_policy(policy_policy, "Optimal Policy from Policy Iteration")
