import numpy as np

def value_iteration(env, gamma=0.99, theta=1e-8):
    states = get_all_states(env)
    V = {state: 0 for state in states}
    
    while True:
        delta = 0
        for state in states:
            v = V[state]
            max_q = float('-inf')
            for action in Action:
                q = 0
                for next_state in states:
                    p = get_transition_prob(state, action, next_state, env)
                    r = get_reward(state, action, next_state, env)
                    q += p * (r + gamma * V[next_state])
                max_q = max(max_q, q)
            V[state] = max_q
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    
    policy = {}
    for state in states:
        best_action = None
        max_q = float('-inf')
        for action in Action:
            q = 0
            for next_state in states:
                p = get_transition_prob(state, action, next_state, env)
                r = get_reward(state, action, next_state, env)
                q += p * (r + gamma * V[next_state])
            if q > max_q:
                max_q = q
                best_action = action
        policy[state] = best_action
    
    return V, policy

# Run Value Iteration
env = SokobanEnv()
V, policy = value_iteration(env)

print("Value Function:")
for state, value in V.items():
    print(f"State {state}: {value}")

print("\nOptimal Policy:")
for state, action in policy.items():
    print(f"State {state}: {action}")