# main.py

from tsp_env import TSPEnv
from factory import create_dynamic_programming_solver, create_monte_carlo_solver, create_monte_carlo_epsilon_greedy_solver
import numpy as np


def main():
    num_cities = 6
    num_episodes = 100

    env = TSPEnv(num_cities, seed=42)

    # Create solvers
    dp_solver = create_dynamic_programming_solver(env)
    mc_solver = create_monte_carlo_solver(env, num_episodes=10000, method="first_visit")
    mc_epsilon_greedy_solver = create_monte_carlo_epsilon_greedy_solver(env, num_episodes=10000, epsilon=0.1, method="first_visit")

    solvers = {
        "Dynamic Programming": dp_solver,
        "Monte Carlo": mc_solver,
        "Monte Carlo Epsilon-Greedy": mc_epsilon_greedy_solver
    }

    total_performance = {}

    for solver_name, solver in solvers.items():
        print(f"\nTesting {solver_name}:")
        episode_returns = []

        for episode in range(num_episodes):
            observation = env.reset()
            done = False
            total_reward = 0
            current_city, visited_cities = observation

            while not done:
                action = solver.get_action(current_city, visited_cities)
                observation, reward, done, _, _ = env.step(action)
                total_reward += reward
                current_city, visited_cities = observation

            episode_returns.append(total_reward)
            print(f"Episode {episode}: Total Reward = {total_reward}")

        avg_return = np.mean(episode_returns)
        print(f"Average return over {num_episodes} episodes: {avg_return}")
        total_performance[solver_name] = avg_return

    print("\nPerformance Comparison:")
    for solver_name, avg_return in total_performance.items():
        print(f"{solver_name} Average Return: {avg_return}")


if __name__ == "__main__":
    main()
