import numpy as np

# Parameters
gamma = 0.9  # Discount factor
max_iterations = 200  # Maximum number of iterations
epsilon = 1e-6  # Convergence threshold
# States
states = [0, 1, 2]  # hostel, academic building, canteen
# Actions
actions = [0, 1]  # 0: study, 1: eat

# Reward function R(s)
rewards = np.array([-1, 3, 1])  # Example rewards for the states

# Transition probabilities P[s][a][s'] gives the probability of transitioning to state s' from state s with action a
transitions = np.array([
    # From state 0
    [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],  # Actions study, eat
    # From state 1
    [[0.0, 0.7, 0.3], [0.0, 0.2, 0.8]],
    # From state 2
    [[0.3, 0.6, 0.1], [0.0, 0.0, 1.0]]
])

# Initialize policy arbitrarily
policy = np.zeros(len(states), dtype=int)

# Initialize value function V(s) arbitrarily
V = np.zeros(len(states))


def policy_evaluation(policy):
    iteration = 0
    while iteration < max_iterations:
        delta = 0
        new_V = np.copy(V)

        for s in states:
            action = policy[s]
            new_V[s] = rewards[s] + gamma * sum([transitions[s][action][s_prime] * V[s_prime] for s_prime in states])
            delta = max(delta, np.abs(new_V[s] - V[s]))

        V[:] = new_V
        iteration += 1

        if delta < epsilon:
            break

    return V


def policy_improvement():
    policy_stable = True

    for s in states:
        old_action = policy[s]

        # Compute the value of each action
        action_values = []
        for a in actions:
            action_value = sum([transitions[s][a][s_prime] * V[s_prime] for s_prime in states])
            action_values.append(action_value)

        # Choose the action that maximizes the expected value
        best_action = np.argmax([rewards[s] + gamma * action_values[a] for a in actions])

        policy[s] = best_action

        if old_action != best_action:
            policy_stable = False

    return policy_stable


def policy_iteration():
    iteration = 0
    while iteration < max_iterations:
        # Policy evaluation
        policy_evaluation(policy)

        # Policy improvement
        policy_stable = policy_improvement()

        iteration += 1

        if policy_stable:
            break

    return policy, V, iteration


# Run Policy Iteration
optimal_policy, optimal_values, iterations = policy_iteration()

# Print results
print(f"Optimal Policy: {optimal_policy} (0: study, 1: eat)")
print(f"Optimal Values for hostel, ab, canteen: {optimal_values}")
# print(f"Converged after {iterations} iterations.")
