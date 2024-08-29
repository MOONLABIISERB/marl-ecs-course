import numpy as np
import matplotlib.pyplot as plt

grid_size = 9
start_state = (0, 0)
goal_state = (8, 8)
obstacles = [(1, 3), (2, 3), (3, 3), (3, 2), (3, 1),
             (5, 5), (6, 5), (7, 5), (8, 5), (5, 6), (5, 7), (5, 8)]
in_portal = (2, 2)
out_portal = (6, 6)
actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
gamma = 0.9

def is_valid(s):
    return 0 <= s[0] < grid_size and 0 <= s[1] < grid_size and s not in obstacles

def next_state(s, a):
    state = (s[0] + a[0], s[1] + a[1])
    if not is_valid(state):
        return s
    if state == in_portal:
        return out_portal
    return state

def value_iteration():
    V = np.zeros((grid_size, grid_size))
    policy = np.zeros((grid_size, grid_size, 2), dtype=int)

    while True:
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                s = (i, j)
                if s == goal_state or s in obstacles:
                    continue
                old_v = V[i, j]
                Q = np.zeros(len(actions))
                for idx, a in enumerate(actions):
                    next_s = next_state(s, a)
                    reward = 1 if next_s == goal_state else 0
                    Q[idx] = reward + gamma * V[next_s[0], next_s[1]]
                V[i, j] = np.max(Q)
                policy[i, j] = actions[np.argmax(Q)]
                delta = max(delta, abs(old_v - V[i, j]))
        if delta < 1e-4:
            break
    
    return policy, V

def policy_iteration():
    policy = np.zeros((grid_size, grid_size, 2), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) != goal_state and (i, j) not in obstacles:
                policy[i, j] = actions[np.random.choice(len(actions))]

    V = np.zeros((grid_size, grid_size))

    while True:
        while True:
            delta = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    s = (i, j)
                    if s == goal_state or s in obstacles:
                        continue
                    a = tuple(policy[i, j])
                    next_s = next_state(s, a)
                    reward = 1 if next_s == goal_state else 0
                    old_v = V[i, j]
                    V[i, j] = reward + gamma * V[next_s[0], next_s[1]]
                    delta = max(delta, abs(old_v - V[i, j]))
            if delta < 1e-4:
                break

        stable = True
        for i in range(grid_size):
            for j in range(grid_size):
                s = (i, j)
                if s == goal_state or s in obstacles:
                    continue
                old_action = tuple(policy[i, j])
                Q = np.zeros(len(actions))
                for idx, a in enumerate(actions):
                    next_s = next_state(s, a)
                    reward = 1 if next_s == goal_state else 0
                    Q[idx] = reward + gamma * V[next_s[0], next_s[1]]
                new_action = actions[np.argmax(Q)]
                policy[i, j] = new_action
                if old_action != new_action:
                    stable = False
        if stable:
            break
    
    return policy, V

def plot_policy(policy, title):
    plt.figure(figsize=(8, 8))
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)

    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) in {goal_state, in_portal} or (i, j) in obstacles:
                continue
            action = policy[i, j]
            U[i, j] = action[1]
            V[i, j] = action[0]

    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='black')

    for x in range(grid_size + 1):
        plt.axhline(x - 0.5, color='black', linewidth=1)
    for y in range(grid_size + 1):
        plt.axvline(y - 0.5, color='black', linewidth=1)

    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.grid(False)
    plt.title(title, fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.plot(goal_state[1], goal_state[0], 'r*', markersize=15)
    plt.text(goal_state[1], goal_state[0], 'Goal', fontsize=12, ha='center', va='center')

    plt.plot(start_state[1], start_state[0], 'go', markersize=10)
    plt.text(start_state[1], start_state[0], 'Start', fontsize=12, ha='center', va='center')

    plt.plot(in_portal[1], in_portal[0], 'bs', markersize=10)
    plt.text(in_portal[1], in_portal[0], 'IN', fontsize=10, ha='center', va='center', color='white')
    plt.plot(out_portal[1], out_portal[0], 'bs', markersize=10)
    plt.text(out_portal[1], out_portal[0], 'OUT', fontsize=10, ha='center', va='center', color='white')

    plt.show()

policy_from_value, _ = value_iteration()
plot_policy(policy_from_value, "Optimal Policy - Value Iteration")

policy_from_policy, _ = policy_iteration()
plot_policy(policy_from_policy, "Optimal Policy - Policy Iteration")
