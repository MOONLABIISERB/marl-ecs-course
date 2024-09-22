import numpy as np

# Define the MDP parameters
factor = 0.9

# Define the state space, action space, transition probabilities, and rewards
states = ['Hostel', 'Academic Building', 'Canteen']
actions = ['Attend Class', 'Eat Food']
transition_prob = {
    ('Hostel', 'Attend Class', 'Academic Building'): 0.5,
    ('Hostel', 'Attend Class', 'Hostel'): 0.5,
    ('Hostel', 'Eat Food', 'Canteen'): 1.0,
    ('Academic Building', 'Attend Class', 'Academic Building'): 0.7,
    ('Academic Building', 'Attend Class', 'Canteen'): 0.3,
    ('Academic Building', 'Eat Food', 'Canteen'): 0.8,
    ('Academic Building', 'Eat Food', 'Academic Building'): 0.2,
    ('Canteen', 'Attend Class', 'Academic Building'): 0.6,
    ('Canteen', 'Attend Class', 'Hostel'): 0.3,
    ('Canteen', 'Attend Class', 'Canteen'): 0.1,
    ('Canteen', 'Eat Food', 'Canteen'): 1.0
}
rewards = {
    ('Hostel', 'Eat Food', 'Canteen'): 1,
    ('Academic Building', 'Attend Class', 'Academic Building'): 3,
    ('Academic Building', 'Attend Class', 'Canteen'): 1,
    ('Canteen', 'Attend Class', 'Academic Building'): 3,
    ('Canteen', 'Attend Class', 'Canteen'): 1
}

# Perform value iteration
def value_iteration():
    V = {s: 0 for s in states}
    while True:
        V_new = {}
        for s in states:
            max_value = float('-inf')
            for a in actions:
                value = sum(transition_prob.get((s, a, s_next), 0) * (rewards.get((s, a, s_next), 0) + factor * V[s_next]) for s_next in states)
                if value > max_value:
                    max_value = value
            V_new[s] = max_value
        if max(abs(V_new[s] - V[s]) for s in states) < 1e-6:
            break
        V = V_new
    return V

# Perform policy iteration
def policy_iteration():
    policy = {s: actions[0] for s in states}
    while True:
        V = {s: 0 for s in states}
        for _ in range(1000):  # Number of policy evaluation iterations
            for s in states:
                a = policy[s]
                V[s] = sum(transition_prob.get((s, a, s_next), 0) * (rewards.get((s, a, s_next), 0) + factor * V[s_next]) for s_next in states)
        policy_stable = True
        for s in states:
            max_value = float('-inf')
            best_action = None
            for a in actions:
                value = sum(transition_prob.get((s, a, s_next), 0) * (rewards.get((s, a, s_next), 0) + factor * V[s_next]) for s_next in states)
                if value > max_value:
                    max_value = value
                    best_action = a
            if best_action != policy[s]:
                policy_stable = False
            policy[s] = best_action
        if policy_stable:
            break
    return policy

# Perform value iteration and policy iteration
optimal_value = value_iteration()
optimal_policy = policy_iteration()

print("Optimal Value Function:")
print(optimal_value)
print("\nOptimal Policy:")
print(optimal_policy)
