import time

def evaluate_policy(env, policy, num_episodes=1000):
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy[state]
            state, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / num_episodes

# Compare Value Iteration and Monte Carlo
env = SokobanEnv()

# Value Iteration
start_time = time.time()
V, vi_policy = value_iteration(env)
vi_time = time.time() - start_time
vi_performance = evaluate_policy(env, vi_policy)

# Monte Carlo
start_time = time.time()
Q, mc_policy = monte_carlo(env)
mc_time = time.time() - start_time
mc_performance = evaluate_policy(env, mc_policy)

print("Value Iteration:")
print(f"Time: {vi_time:.2f} seconds")
print(f"Average Reward: {vi_performance:.2f}")

print("\nMonte Carlo:")
print(f"Time: {mc_time:.2f} seconds")
print(f"Average Reward: {mc_performance:.2f}")