import numpy as np

# States
states = ["Hostel", "Academic_Building", "Canteen"]
# Actions
actions = ["Class", "Eat"]
# Transition Probabilities and Rewards based on the provided MDP
transition_probs = {
    "Hostel": {
        "Class": [("Hostel", 0.5), ("Academic_Building", 0.5)],
        "Eat": [("Canteen", 1.0)]
    },
    "Academic_Building": {
        "Class": [("Academic_Building", 0.7), ("Canteen", 0.3)],
        "Eat": [("Canteen", 0.8), ("Academic_Building", 0.2)]
    },
    "Canteen": {
        "Class": [("Academic_Building", 0.6), ("Hostel", 0.3), ("Canteen", 0.1)],
        "Eat": [("Canteen", 1.0)]
    }
}
# Rewards for each state
rewards = {
    "Hostel": -1,
    "Academic_Building": 3,
    "Canteen": 1
}
discount_factor = 0.9

# Value Iteration Function
def value_iteration(states, actions, transition_probs, rewards, discount_factor, threshold=1e-5):
    V = {state: 0 for state in states}
    policy = {state: None for state in states}
    
    while True:
        delta = 0
        for state in states:
            action_values = []
            for action in actions:
                action_value = rewards[state] + discount_factor * sum(prob * V[next_state] 
                                   for next_state, prob in transition_probs[state].get(action, []))
                action_values.append((action_value, action))
            
            max_value, best_action = max(action_values, default=(V[state], None))
            delta = max(delta, abs(V[state] - max_value))
            V[state] = max_value
            policy[state] = best_action
        
        if delta < threshold:
            break
    
    return V, policy

# Policy Evaluation Function
def policy_evaluation(states, policy, transition_probs, rewards, discount_factor, threshold=1e-5):
    V = {state: 0 for state in states}
    
    while True:
        delta = 0
        for state in states:
            action = policy[state]
            if action is None:
                continue
            value = rewards[state] + discount_factor * sum(prob * V[next_state] 
                        for next_state, prob in transition_probs[state].get(action, []))
            delta = max(delta, abs(V[state] - value))
            V[state] = value
        
        if delta < threshold:
            break
    
    return V

# Policy Iteration Function
def policy_iteration(states, actions, transition_probs, rewards, discount_factor):
    policy = {state: np.random.choice(actions) for state in states}
    
    while True:
        V = policy_evaluation(states, policy, transition_probs, rewards, discount_factor)
        
        policy_stable = True
        for state in states:
            old_action = policy[state]
            action_values = []
            for action in actions:
                action_value = rewards[state] + discount_factor * sum(prob * V[next_state] 
                                   for next_state, prob in transition_probs[state].get(action, []))
                action_values.append((action_value, action))
            
            _, best_action = max(action_values, default=(V[state], None))
            policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy

# Running Value Iteration
optimal_values, optimal_policy_value_iteration = value_iteration(states, actions, transition_probs, rewards, discount_factor)
print("Optimal Values from Value Iteration:", optimal_values)
print("Optimal Policy from Value Iteration:", optimal_policy_value_iteration)

# Running Policy Iteration
optimal_policy_policy_iteration = policy_iteration(states, actions, transition_probs, rewards, discount_factor)
print("Optimal Policy from Policy Iteration:", optimal_policy_policy_iteration)

