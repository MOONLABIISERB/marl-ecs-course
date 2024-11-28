"""
##########################

Author : Shirish Shekhar Jha
Roll : 2410705

##########################

"""


import numpy as np

# states
states = ['H', 'A', 'C'] # H means hostel, A means academic building, and C means Canteen

#  actions
actions = ['Eat', 'Class']

# transition probabilities
transition_probabilities = {
    'H': {
        'Eat': {'H': 0.0, 'A': 0.0, 'C': 1.0},
        'Class': {'H': 0.5, 'A': 0.5, 'C': 0.0}
    },
    'A': {
        'Eat': {'H': 0.0, 'A': 0.2, 'C': 0.8},
        'Class': {'H': 0.0, 'A': 0.7, 'C': 0.3}
    },
    'C': {
        'Eat': {'H': 0.0, 'A': 0.0, 'C': 1.0},
        'Class': {'H': 0.3, 'A': 0.6, 'C': 0.1}
    }
}

# rewards
rewards = {
    'H': -1,
    'A': 3,
    'C': 1
}

# Value Iteration
def value_iteration(states, actions, transition_probabilities, rewards, gamma=0.9, theta=1e-6):
    V = {s: 0 for s in states}
    policy = {s: None for s in states}
    
    while True:
        delta = 0
        for s in states:
            old_v = V[s]
            action_values = {}
            for a in actions:
                action_values[a] = sum([transition_probabilities[s][a][s_next] * (rewards[s_next] + gamma * V[s_next]) for s_next in states])
            best_action = max(action_values, key=action_values.get)
            V[s] = action_values[best_action]
            policy[s] = best_action
            delta = max(delta, abs(old_v - V[s]))
        if delta < theta:
            break
    return V, policy

# Policy Iteration
def policy_iteration(states, actions, transition_probabilities, rewards, gamma=0.9):
    policy = {s: np.random.choice(actions) for s in states}
    V = {s: 0 for s in states}
    
    def policy_evaluation(policy, V, gamma=0.9, theta=1e-6):
        while True:
            delta = 0
            for s in states:
                old_v = V[s]
                a = policy[s]
                V[s] = sum([transition_probabilities[s][a][s_next] * (rewards[s_next] + gamma * V[s_next]) for s_next in states])
                delta = max(delta, abs(old_v - V[s]))
            if delta < theta:
                break
        return V

    while True:
        stable = True
        V = policy_evaluation(policy, V, gamma)
        for s in states:
            old_action = policy[s]
            action_values = {}
            for a in actions:
                action_values[a] = sum([transition_probabilities[s][a][s_next] * (rewards[s_next] + gamma * V[s_next]) for s_next in states])
            best_action = max(action_values, key=action_values.get)
            if best_action != old_action:
                stable = False
                policy[s] = best_action
        if stable:
            break
    return V, policy

# Value Iteration
optimal_values_vi, optimal_policy_vi = value_iteration(states, actions, transition_probabilities, rewards)
print("Value Iteration - Optimal Values:", optimal_values_vi)
print("Value Iteration - Optimal Policy:", optimal_policy_vi)

# Policy Iteration
optimal_values_pi, optimal_policy_pi = policy_iteration(states, actions, transition_probabilities, rewards)
print("Policy Iteration - Optimal Values:", optimal_values_pi)
print("Policy Iteration - Optimal Policy:", optimal_policy_pi)

