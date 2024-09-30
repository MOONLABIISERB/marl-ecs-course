from tsp import TSP
from typing import Dict, List, Optional, Tuple
import numpy as np

if __name__ == "__main__":
    num_targets = 5
    max_episodes = 10000
    max_steps = 10

    env = TSP(num_targets)
    obs, _ = env.reset()
    episode_returns = []
    policy = {}
    Q_values = {}
    Returns = {}

    gamma = 0.9  # Discount factor

    for episode in range(max_episodes):
        G = 0
        episode_data = []

        obs_, _ = env.reset()
        action = env.action_space.sample()

        for step in range(max_steps):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_data.append((obs_, action, reward))
            obs_ = next_obs
            action = policy.get(env.loc, env.action_space.sample())

            if done:
                break

        G = 0
        visited_state_actions = set()

        for obs_, action, reward in reversed(episode_data):
            G = reward + gamma * G

            state_action_key = (env.loc, action)
            if state_action_key not in visited_state_actions:
                visited_state_actions.add(state_action_key)

                if state_action_key not in Returns:
                    Returns[state_action_key] = []
                Returns[state_action_key].append(G)

                Q_values[state_action_key] = np.mean(Returns[state_action_key])

                best_action = max(
                    [a for a in range(env.action_space.n) if a != state_action_key[0]],
                    key=lambda a: Q_values.get((env.loc, a), float('-inf'))
                )
                policy[env.loc] = best_action

        episode_returns.append(G)
        print(f"Episode {episode}: Return = {G}")

    print(f"Average return: {np.mean(episode_returns)}")
    print(policy)

    # Testing the trained policy
    env.reset()
    print()
    for i in range(num_targets):
        action = policy[env.loc]
        obs_, reward, terminated, truncated, info = env.step(action)
        print(f"Taken action: {action}")