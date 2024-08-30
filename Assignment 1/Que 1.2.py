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

policy = {s: np.random.choice(actions) for s in states}

gamma = 0.9

def policy_evaluation(policy, transition_probabilities, gamma, theta=1e-6):
    V = {s: 0 for s in states}
    while True:
        delta = 0
        for state in states:
            v = V[state]
            action = policy[state]
            V[state] = sum(prob * (reward + gamma * V[next_state])
                           for next_state, prob, reward in transition_probabilities[state][action])
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def policy_improvement(V, transition_probabilities, gamma):
    policy_stable = True
    for state in states:
        old_action = policy[state]
        action_values = {}
        for action in actions:
            action_value = sum(prob * (reward + gamma * V[next_state])
                               for next_state, prob, reward in transition_probabilities[state][action])
            action_values[action] = action_value
        policy[state] = max(action_values, key=action_values.get)
        if old_action != policy[state]:
            policy_stable = False
    return policy_stable

def policy_iteration(transition_probabilities, gamma):
    while True:
        V = policy_evaluation(policy, transition_probabilities, gamma)
        policy_stable = policy_improvement(V, transition_probabilities, gamma)
        if policy_stable:
            break
    return policy

optimal_policy = policy_iteration(transition_probabilities, gamma)

print("Optimal Policy:")
for state in states:
    print(f"State {state}: {optimal_policy[state]}")
