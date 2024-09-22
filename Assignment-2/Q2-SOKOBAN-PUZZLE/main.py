# main.py

import time
import numpy as np
from sokoban_env import SokobanEnv
from dynamic_programming import DynamicProgrammingAgent
from monte_carlo import MonteCarloAgent


def run_episode(env, agent, max_steps=100, render=False):
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done and steps < max_steps:
        current_state = env.get_state()
        action = agent.select_action(current_state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        if render:
            env.render()
    return total_reward, steps


def evaluate_agent(env, agent, num_episodes=100):
    rewards = []
    steps_list = []
    for _ in range(num_episodes):
        total_reward, steps = run_episode(env, agent)
        rewards.append(total_reward)
        steps_list.append(steps)
    return np.mean(rewards), np.mean(steps_list)


def main():
    env = SokobanEnv()

    print("Training Dynamic Programming Agent...")
    start_time = time.time()
    dp_agent = DynamicProgrammingAgent(env)
    dp_agent.value_iteration()
    dp_training_time = time.time() - start_time
    print(f"Dynamic Programming training completed in {dp_training_time:.2f} seconds")

    print("\nTraining Monte Carlo Agent...")
    start_time = time.time()
    mc_agent = MonteCarloAgent(env)
    mc_agent.train(num_episodes=10000)
    mc_training_time = time.time() - start_time
    print(f"Monte Carlo training completed in {mc_training_time:.2f} seconds")

    print("\nEvaluating Dynamic Programming Agent...")
    dp_avg_reward, dp_avg_steps = evaluate_agent(env, dp_agent)
    print(f"Dynamic Programming - Average Reward: {dp_avg_reward:.2f}, Average Steps: {dp_avg_steps:.2f}")

    print("\nEvaluating Monte Carlo Agent...")
    mc_avg_reward, mc_avg_steps = evaluate_agent(env, mc_agent)
    print(f"Monte Carlo - Average Reward: {mc_avg_reward:.2f}, Average Steps: {mc_avg_steps:.2f}")

    print("\nComparison:")
    print(f"Training Time - DP: {dp_training_time:.2f}s, MC: {mc_training_time:.2f}s")
    print(f"Average Reward - DP: {dp_avg_reward:.2f}, MC: {mc_avg_reward:.2f}")
    print(f"Average Steps - DP: {dp_avg_steps:.2f}, MC: {mc_avg_steps:.2f}")


if __name__ == "__main__":
    main()
