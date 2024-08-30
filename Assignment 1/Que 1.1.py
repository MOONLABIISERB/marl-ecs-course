import numpy as np

states = ['H', 'A', 'C']
actions = ['Attend Class', 'Eat']

transition_probabilities = {
    'H': {
        'Attend Class': [('A', 0.5, 3), ('H', 0.5, -1)],
        'Eat': [('C', 1.0, 1)]
    },
    'A': {
        'Attend Class': [('A', 0.7, 3), ('C', 0.3, 1)],
        'Eat': [('C', 0.8, 1), ('A', 0.2, 3)]
    },
    'C': {
        'Attend Class': [('A', 0.6, 3), ('H', 0.3, -1), ('C', 0.1, 1)],
        'Eat': [('C', 1.0, 1)]
    }
}

V = {s: 0 for s in states}

gamma = 0.9

def value_iteration(transition_probabilities, V, gamma, theta=1e-6):
    while True:
        delta = 0
        for state in states:
            v = V[state]
            action_values = []
            for action in actions:
                action_value = 0
                for next_state, prob, reward in transition_probabilities[state][action]:
                    action_value += prob * (reward + gamma * V[next_state])
                action_values.append(action_value)
            V[state] = max(action_values)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def extract_policy(transition_probabilities, V, gamma):
    policy = {}
    for state in states:
        action_values = {}
        for action in actions:
            action_value = 0
            for next_state, prob, reward in transition_probabilities[state][action]:
                action_value += prob * (reward + gamma * V[next_state])
            action_values[action] = action_value
        policy[state] = max(action_values, key=action_values.get)
    return policy


optimal_values = value_iteration(transition_probabilities, V, gamma)
optimal_policy = extract_policy(transition_probabilities, optimal_values, gamma)

print("Optimal Values for each state:")
for state in states:
    print(f"State {state}: {optimal_values[state]:.2f}")

print("\nOptimal Policy:")
for state in states:
    print(f"State {state}: {optimal_policy[state]}")
