import numpy as np
import time
from tqdm import tqdm
from sokoban import SokobanEnv
from dp import DynamicProgramming
from mc import MonteCarlo


def run_episode(env, policy, save_frames=False):
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    frames = []
    while not done and steps < 100:
        action = policy.get_action(env.get_state())
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        if save_frames:
            frames.append(env.render(mode="rgb_array"))
    return total_reward, steps, frames


def evaluate_policy(env, policy, num_episodes=100):
    total_rewards = []
    total_steps = []
    for _ in range(num_episodes):
        reward, steps, _ = run_episode(env, policy)
        total_rewards.append(reward)
        total_steps.append(steps)
    return np.mean(total_rewards), np.mean(total_steps)


def main():
    env = SokobanEnv()

    print(
        "Initializing and training Dynamic Programming method"
    )
    start_time = time.time()
    dp_solver = DynamicProgramming(env)
    dp_solver.value_iteration()
    dp_training_time = time.time() - start_time
    print(f"DP training completed in {dp_training_time:.2f} seconds")

    print("Initializing and training Monte Carlo method")
    start_time = time.time()
    mc_solver = MonteCarlo(env)
    mc_solver.train(num_episodes=10000)
    mc_training_time = time.time() - start_time
    print(f"MC training completed in {mc_training_time:.2f} seconds")

    print("Evaluating Dynamic Programming policy")
    dp_avg_reward, dp_avg_steps = evaluate_policy(env, dp_solver)
    print(f"DP Average Reward: {dp_avg_reward:.2f}")
    print(f"DP Average Steps: {dp_avg_steps:.2f}")

    print("Evaluating Monte Carlo policy")
    mc_avg_reward, mc_avg_steps = evaluate_policy(env, mc_solver)
    print(f"MC Average Reward: {mc_avg_reward:.2f}")
    print(f"MC Average Steps: {mc_avg_steps:.2f}")

if __name__ == "__main__":
    main()
