import numpy as np
import random
from itertools import product
import matplotlib.pyplot as plt
from test import ModTSP  # Importing the ModTSP class


def create_q_table(num_targets):
    """Creates a Q-table for the environment."""
    Q = {}
    state_range = range(num_targets)
    visited_range = [0, 1]
    binary_combinations = list(product(visited_range, repeat=num_targets))
    action_range = range(num_targets)
    for state in state_range:
        for binary_combination in binary_combinations:
            a = (state,) + binary_combination
            for b in action_range:
                Q[(a, b)] = 0
    return Q


def get_max(Q, a, num_targets):
    max_value = -float('inf')
    for b in range(num_targets):
        if (tuple(a), b) in Q:
            max_value = max(max_value, Q[(tuple(a), b)])
    return max_value


def best_action(Q, a, num_targets):
    max_value = -float('inf')
    best_action = None
    for b in range(num_targets):
        if (tuple(a), b) in Q:
            current_value = Q[(tuple(a), b)]
            if current_value > max_value:
                max_value = current_value
                best_action = b
    return best_action


def epsilon_greedy(epsilon, Q, state, env, num_targets):
    a = random.random()
    if a < epsilon:
        action = env.action_space.sample()
    else:
        action = best_action(Q, state, num_targets)
    return action


def main():
    num_targets = 10
    env = ModTSP(num_targets)
    Q = create_q_table(num_targets)
    gamma = 0.7
    alpha = 0.2
    max_epsilon = 0.5
    min_epsilon = 0.001
    decay_rate = 0.0007
    episodes = []
    max_steps = 100

    ep_rets = []
    avg_rets = []  # Running average rewards

    for ep in range(40000):
        ret = 0
        obs, _ = env.reset()
        state = obs[:11].astype(int)
        state = [int(x) for x in state]
        visited_path = []
        for _ in range(max_steps):
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * ep)
            if visited_path == []:
                action = env.action_space.sample()
            else:
                action = epsilon_greedy(epsilon, Q, state, env, num_targets)
            obs_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = obs_[:11].astype(int)
            next_state = [int(x) for x in next_state]

            Q[tuple(state), action] += alpha * (
                reward + gamma * get_max(Q, next_state, num_targets) - Q[tuple(state), action]
            )
            ret += reward
            visited_path.append(next_state[0])
            if done:
                break
            state = next_state

        print(f"Episode {ep} : {ret}  epsilon: {epsilon}")
        print(f"Visited Path: {visited_path}")
        print(f"Unique points: {len(np.unique(visited_path))}")
        ep_rets.append(ret)
        episodes.append(ep)
        avg_rets.append(np.mean(ep_rets))

    np.save("q_table.npy", Q)
    # Plotting the rewards
    plt.plot(episodes, ep_rets, label='Total Reward per Episode')
    plt.plot(episodes, avg_rets, label='Running Average Reward', color='orange', linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episodes vs Total Reward and Average Reward')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
