import numpy as np
import time
from tqdm import tqdm
from sokoban import SokobanEnv
from dynamic_problem import DynamicProgrammingSolver
from mc_solver import MonteCarloSolver


def play_episode(env, strategy, record_frames=False):
    """Play a single episode following the given strategy."""
    state, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    frames = []

    # Loop until the episode is finished or we hit the step limit
    while not done and step_count < 100:
        action = strategy.get_action(env.get_state())
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

        if record_frames:
            frames.append(env.render(mode="rgb_array"))
    
    return total_reward, step_count, frames


def evaluate_strategy(env, strategy, episodes=100):
    """Evaluate the given strategy over a number of episodes."""
    rewards = []
    steps = []
    for _ in range(episodes):
        reward, steps_taken, _ = play_episode(env, strategy)
        rewards.append(reward)
        steps.append(steps_taken)
    
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)

    return avg_reward, avg_steps


def main():
    env = SokobanEnv()

    # Train Dynamic Programming Solver
    print("Training Dynamic Programming solver... This might take a while, please be patient.")
    start_time = time.time()
    dp_solver = DynamicProgrammingSolver(env)
    dp_solver.train()
    dp_training_duration = time.time() - start_time
    print(f"Dynamic Programming training completed in {dp_training_duration:.2f} seconds.")

    # Train Monte Carlo Solver
    print("\nTraining Monte Carlo solver...")
    start_time = time.time()
    mc_solver = MonteCarloSolver(env)
    mc_solver.train(episodes=10000)
    mc_training_duration = time.time() - start_time
    print(f"Monte Carlo training completed in {mc_training_duration:.2f} seconds.")

    # Evaluate Dynamic Programming Strategy
    print("\nEvaluating the Dynamic Programming strategy...")
    dp_avg_reward, dp_avg_steps = evaluate_strategy(env, dp_solver)
    print(f"Dynamic Programming - Average Reward: {dp_avg_reward:.2f}, Average Steps: {dp_avg_steps:.2f}")

    # Evaluate Monte Carlo Strategy
    print("\nEvaluating the Monte Carlo strategy...")
    mc_avg_reward, mc_avg_steps = evaluate_strategy(env, mc_solver)
    print(f"Monte Carlo - Average Reward: {mc_avg_reward:.2f}, Average Steps: {mc_avg_steps:.2f}")

    # Compare Results
    print("\nComparison of Solvers:")
    print(f"Training Time - Dynamic Programming: {dp_training_duration:.2f}s, Monte Carlo: {mc_training_duration:.2f}s")
    print(f"Average Reward - Dynamic Programming: {dp_avg_reward:.2f}, Monte Carlo: {mc_avg_reward:.2f}")
    print(f"Average Steps - Dynamic Programming: {dp_avg_steps:.2f}, Monte Carlo: {mc_avg_steps:.2f}")

    # Optionally, create GIFs for policy visualization (commented out)
    # print("\nSaving GIF for Dynamic Programming strategy...")
    # _, _, dp_frames = play_episode(env, dp_solver, record_frames=True)
    # save_gif(dp_frames, "dp_strategy.gif")

    # print("\nSaving GIF for Monte Carlo strategy...")
    # _, _, mc_frames = play_episode(env, mc_solver, record_frames=True)
    # save_gif(mc_frames, "mc_strategy.gif")

    print("\nGIFs saved successfully")


if __name__ == "__main__":
    main()
