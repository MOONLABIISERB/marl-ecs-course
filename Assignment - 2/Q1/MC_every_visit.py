import numpy as np
from typing import Dict, List, Optional, Tuple
from tsp import TSP




if __name__ == "__main__":
    num_targets = 5
    max_episodes = 10000
    max_steps = 55

    env = TSP(num_targets)
    obs, _ = env.reset()
    episode_returns = []
    policy = {}
    Q_values = {}
    Returns = {}

    discount_factor = 0.9  # Discount rate for rewards

    # Training phase
    for episode in range(max_episodes):
        total_return = 0
        episode_data = []
        obs_, _ = env.reset()
        action = env.action_space.sample()
        state_action_key = (obs_, action)

        for step in range(max_steps):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_data.append((obs_, action, reward))

            obs_ = next_obs
            action = policy.get(env.loc, env.action_space.sample())

            if done:
                break

        G = 0

        # Every-visit Monte Carlo update
        for obs_, action, reward in reversed(episode_data):
            G = reward + discount_factor * G
            state_action_key = (env.loc, action)

            if state_action_key not in Returns:
                Returns[state_action_key] = []
            Returns[state_action_key].append(G)

            Q_values[state_action_key] = np.mean(Returns[state_action_key])

            # Update policy based on the best action
            best_action = max(
                [a for a in range(env.action_space.n) if a != state_action_key[0]],
                key=lambda a: Q_values.get((env.loc, a), float('-inf'))
            )
            policy[env.loc] = best_action

        episode_returns.append(G)
        print(f"Episode {episode}: {G}")

    print(f"Average return: {np.mean(episode_returns)}")

    # Testing phase
    print()
    for i in range(num_targets):
        action = policy[env.loc]
        obs_, reward, terminated, truncated, info = env.step(action)
        print(f"Taken action {action}")