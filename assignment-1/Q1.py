import numpy as np

states = ['H', 'A', 'C']
actions = ['Attend_Class', 'hungry']

rewards = {
    'H': -1,
    'A': 3,
    'C': 1
}

transition_probs = {
    ('H', 'Attend_Class'): {'H': 0.5, 'A': 0.5},
    ('H', 'hungry'): {'C': 1.0},
    ('A', 'Attend_Class'): {'A': 0.7, 'C': 0.3},
    ('A', 'hungry'): {'C': 0.8, 'A': 0.2},
    ('C', 'Attend_Class'): {'A': 0.6, 'H': 0.3, 'C': 0.1},
    ('C', 'hungry'): {'C': 1.0}
}

gamma = 0.9
threshold = 1e-6

V = np.zeros(len(states))
state_index = {state: idx for idx, state in enumerate(states)}

def get_next_state_values(state, action):

    next_states_probs = transition_probs.get((state, action), {})
    return sum(prob * V[state_index[next_state]] for next_state, prob in next_states_probs.items())

# Value Iteration Algorithm
def value_iteration(states, actions, transition_probs, rewards, gamma, threshold):
    V = np.zeros(len(states))
    state_index = {state: idx for idx, state in enumerate(states)}
    
    while True:
        delta = 0
        for s in states:
            state_idx = state_index[s]
            old_value = V[state_idx]
            
            # Calculate the maximum value for the current state
            action_values = []
            for a in actions:
                if (s, a) in transition_probs:
                    action_value = sum(
                        prob * (rewards.get(next_state, 0) + gamma * V[state_index[next_state]])
                        for next_state, prob in transition_probs[(s, a)].items()
                    )
                    action_values.append(action_value)
                else:
                    action_values.append(0)
            
            # Update value function
            V[state_idx] = max(action_values)
            delta = max(delta, abs(old_value - V[state_idx]))
        
        # Check convergence 
        if delta < threshold:
            break
    
    # Optimal policy
    policy = np.zeros(len(states), dtype=int)
    for s in states:
        state_idx = state_index[s]
        best_action_value = float('-inf')
        best_action = None
        
        for a in actions:
            if (s, a) in transition_probs:
                action_value = sum(
                    prob * (rewards.get(next_state, 0) + gamma * V[state_index[next_state]])
                    for next_state, prob in transition_probs[(s, a)].items()
                )
                
                if action_value > best_action_value:
                    best_action_value = action_value
                    best_action = a
        
        policy[state_idx] = actions.index(best_action) if best_action else -1
    
    return V, policy

def policy_evaluation(policy, V, gamma, threshold):
    while True:
        delta = 0
        for s in states:
            state_idx = state_index[s]
            a = actions[policy[state_idx]]
            # Compute the value function for the current policy
            new_value = sum(
                prob * (rewards.get(next_state, 0) + gamma * V[state_index[next_state]])
                for next_state, prob in transition_probs.get((s, a), {}).items()
            )
            delta = max(delta, abs(V[state_idx] - new_value))
            V[state_idx] = new_value
        if delta < threshold:
            break

def policy_improvement(V):
    policy_stable = True
    for s in states:
        state_idx = state_index[s]
        old_action = policy[state_idx]
        
        # Find the best action for the current state
        action_values = []
        for a in actions:
            action_value = sum(
                prob * (rewards.get(next_state, 0) + gamma * V[state_index[next_state]])
                for next_state, prob in transition_probs.get((s, a), {}).items()
            )
            action_values.append(action_value)
        
        best_action = np.argmax(action_values)
        policy[state_idx] = best_action
        
        if old_action != policy[state_idx]:
            policy_stable = False
    return policy_stable

def policy_iteration(states, actions, transition_probs, rewards, gamma, threshold):
    global V, policy
    V = np.zeros(len(states))
    policy = np.zeros(len(states), dtype=int)
    
    while True:
        policy_evaluation(policy, V, gamma, threshold)
        stable = policy_improvement(V)
        if stable:
            break
    
    return V, policy

# Value Iteration Implementation
optimal_values1, optimal_policy1 = value_iteration(states, actions, transition_probs, rewards, gamma, threshold)

# Policy Iteration Implementation
optimal_values2, optimal_policy2 = policy_iteration(states, actions, transition_probs, rewards, gamma, threshold)

print("\nAccording to Value Iteration:\n")
print("Optimal Values:")
for state, value in zip(states, optimal_values1):
    print(f"Value of state {state}: {value:.2f}")

print("\nOptimal Policy:")
for state, action_index in zip(states, optimal_policy1):
    print(f"Policy for state {state}: {actions[action_index] if action_index != -1 else 'None'}")

print("\n\nAccording to Policy Iteration:\n")
print("Optimal Values:")
for state, value in zip(states, optimal_values2):
    print(f"Value of state {state}: {value:.2f}")

print("\nOptimal Policy:")
for state, action_idx in zip(states, optimal_policy2):
    print(f"Policy for state {state}: {actions[action_idx]}")

print("\nWhere H = Hostel, A = Academic Building, C = Canteen\n")
