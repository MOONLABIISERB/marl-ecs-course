from tsp_env import TSPEnv
from solvers.dynamic_programming_solver import DynamicProgrammingSolver
from solvers.monte_carlo_solver import MonteCarloSolver
from solvers.monte_carlo_epsilon_greedy_solver import MonteCarloEpsilonGreedySolver


def create_dynamic_programming_solver(env: TSPEnv) -> DynamicProgrammingSolver:
    """Creates and solves a Dynamic Programming solver for the TSP environment."""
    solver = DynamicProgrammingSolver(env.num_cities, env.distance_matrix)
    solver.solve()
    return solver


def create_monte_carlo_solver(env: TSPEnv, num_episodes: int = 10000, method: str = "first_visit") -> MonteCarloSolver:
    """Creates and trains a Monte Carlo solver for the TSP environment."""
    solver = MonteCarloSolver(env.num_cities, env.distance_matrix)
    solver.solve(num_episodes, method)
    return solver


def create_monte_carlo_epsilon_greedy_solver(env: TSPEnv, num_episodes: int = 10000, epsilon: float = 0.1, method: str = "first_visit") -> MonteCarloEpsilonGreedySolver:
    """Creates and trains a Monte Carlo Epsilon-Greedy solver for the TSP environment."""
    solver = MonteCarloEpsilonGreedySolver(env.num_cities, env.distance_matrix, epsilon)
    solver.solve(num_episodes, method)
    return solver
