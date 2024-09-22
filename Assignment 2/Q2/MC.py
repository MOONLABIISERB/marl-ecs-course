import random

def monte_carlo(env, num_episodes=10000, gamma=0.99):
    states = get_all_states(env)
    Q = {(state, action): 0 for state in states for action in Action}
    returns = {(state, action): [] for state in states for action in Action}
    policy = {state: random.choice(list(Action)) for state in states}

    def generate_episode():
        episode = []
        state = random.choice(states)
        env.agent_pos = [state[0], state[1]]
        env.box_pos = [state[2], state[3]]
        env.done = False
        while not env.done:
            action = random.choice(list(Action))  # Exploring starts
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

    for _ in range(num_episodes):
        episode = generate_episode()
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])
                best_action = max(Action, key=lambda a: Q[(state, a)])
                policy[state] = best_action

    return Q, policy

# Run Monte Carlo
env = SokobanEnv()
Q, policy = monte_carlo(env)

print("Q-values:")
for (state, action), value in Q.items():
    print(f"State {state}, Action {action}: {value}")

print("\nOptimal Policy:")
for state, action in policy.items():
    print(f"State {state}: {action}")