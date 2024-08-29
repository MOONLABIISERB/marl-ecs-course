import numpy as np

# Define the states in the MDP
states = ['Hostel', 'Academic Building', 'Canteen']

# Define the actions the student can take
actions = ['Eat', 'Class']

# Define the transition probabilities for each state-action pair
transition_probabilities = {
    'Hostel': {
        'Eat': {'Hostel': 0.0, 'Academic Building': 0.0, 'Canteen': 1.0},
        'Class': {'Hostel': 0.5, 'Academic Building': 0.5, 'Canteen': 0.0}
    },
    'Academic Building': {
        'Eat': {'Hostel': 0.0, 'Academic Building': 0.2, 'Canteen': 0.8},
        'Class': {'Hostel': 0.0, 'Academic Building': 0.7, 'Canteen': 0.3}
    },
    'Canteen': {
        'Eat': {'Hostel': 0.0, 'Academic Building': 0.0, 'Canteen': 1.0},
        'Class': {'Hostel': 0.3, 'Academic Building': 0.6, 'Canteen': 0.1}
    }
}

# Define the rewards for each state
rewards = {
    'Hostel': -1,
    'Academic Building': 3,
    'Canteen': 1
}

# Discount factor (gamma) used in value and policy iteration
gamma = 0.9
theta = 1e-6  # Threshold for convergence

# Function for Value Iteration
def value_iteration(states, actions, transition_probabilities, rewards, gamma=0.9, theta=1e-6):
    # Initialize value function V(s) for all states to zero
    V = {s: 0 for s in states}
    # Initialize an empty policy dictionary
    policy = {s: None for s in states}
    
    while True:
        delta = 0  # Change in value function
        
        # Iterate over each state
        for s in states:
            old_v = V[s]  # Store the current value of the state
            action_values = {}  # Dictionary to store action values
            
            # Iterate over each action available in the current state
            for a in actions:
                # Calculate the action's value based on the transition probabilities and rewards
                action_values[a] = sum([transition_probabilities[s][a][s_next] * (rewards[s_next] + gamma * V[s_next]) for s_next in states])
            
            # Find the action with the highest value and update the value function and policy
            best_action = max(action_values, key=action_values.get)
            V[s] = action_values[best_action]
            policy[s] = best_action
            delta = max(delta, abs(old_v - V[s]))  # Update delta to check for convergence
        
        # Check if the value function has converged
        if delta < theta:
            break
    
    return V, policy

# Function for Policy Evaluation (used in Policy Iteration)
def policy_evaluation(policy, states, transition_probabilities, rewards, V, gamma=0.9, theta=1e-6):
    while True:
        delta = 0  # Change in value function
        
        # Iterate over each state
        for s in states:
            old_v = V[s]  # Store the current value of the state
            a = policy[s]  # Get the current policy's action for the state
            
            # Calculate the value of the state under the current policy
            V[s] = sum([transition_probabilities[s][a][s_next] * (rewards[s_next] + gamma * V[s_next]) for s_next in states])
            delta = max(delta, abs(old_v - V[s]))  # Update delta to check for convergence
        
        # Check if the value function has converged
        if delta < theta:
            break
    
    return V

# Function for Policy Iteration
def policy_iteration(states, actions, transition_probabilities, rewards, gamma=0.9):
    # Initialize a random policy
    policy = {s: np.random.choice(actions) for s in states}
    # Initialize value function V(s) for all states to zero
    V = {s: 0 for s in states}
    
    while True:
        stable = True  # Flag to check if policy is stable
        
        # Evaluate the current policy
        V = policy_evaluation(policy, states, transition_probabilities, rewards, V, gamma)
        
        # Iterate over each state to improve the policy
        for s in states:
            old_action = policy[s]  # Keep track of the old action
            action_values = {}  # Dictionary to store action values
            
            # Iterate over each action to find the best one
            for a in actions:
                # Calculate the action's value
                action_values[a] = sum([transition_probabilities[s][a][s_next] * (rewards[s_next] + gamma * V[s_next]) for s_next in states])
            
            # Find the action with the highest value
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action  # Update the policy
            
            # Check if the policy has changed
            if best_action != old_action:
                stable = False
        
        # If the policy is stable, return the optimal values and policy
        if stable:
            break
    
    return V, policy

# Perform Value Iteration
optimal_values_vi, optimal_policy_vi = value_iteration(states, actions, transition_probabilities, rewards, gamma, theta)
print("Value Iteration - Optimal Values:", optimal_values_vi)
print("Value Iteration - Optimal Policy:", optimal_policy_vi)

# Perform Policy Iteration
optimal_values_pi, optimal_policy_pi = policy_iteration(states, actions, transition_probabilities, rewards, gamma)
print("Policy Iteration - Optimal Values:", optimal_values_pi)
print("Policy Iteration - Optimal Policy:", optimal_policy_pi)
