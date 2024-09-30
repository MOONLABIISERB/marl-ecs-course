"""Environment for Solving the Traveling Salesman Problem (TSP)."""

from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
from dynamic_problem import create_dp_solver, create_mc_solver, create_mc_epsilon_greedy_solver


class TSPEnvironment(gym.Env):
    """Custom RL environment for the Traveling Salesman Problem (TSP).

    The agent's objective is to visit a set of locations while minimizing the total travel distance.
    """

    def __init__(self, target_count: int, grid_size: int = 30, rng_seed: int = None) -> None:
        """Initialize the TSP environment with random target points.

        Args:
            target_count (int): Number of locations the agent needs to visit.
            grid_size (int): The size of the area where the locations are distributed. Defaults to 30.
            rng_seed (int, optional): Seed for reproducibility. Defaults to None.
        """
        super().__init__()
        if rng_seed is not None:
            np.random.seed(seed=rng_seed)

        self.step_count: int = 0
        self.target_count: int = target_count
        self.max_moves: int = target_count
        self.grid_size: int = grid_size

        # Generate random target locations and calculate distances
        self.coordinates: np.ndarray = self._generate_random_targets(self.target_count)
        self.distance_matrix: np.ndarray = self._compute_distance_matrix(self.coordinates)

        # Define observation and action spaces
        self.obs_low = np.concatenate(
            [
                np.array([0], dtype=np.float32),
                np.zeros(self.target_count, dtype=np.float32),
                np.zeros(2 * self.target_count, dtype=np.float32),
            ]
        )
        self.obs_high = np.concatenate(
            [
                np.array([self.target_count], dtype=np.float32),
                2 * self.grid_size * np.ones(self.target_count, dtype=np.float32),
                self.grid_size * np.ones(2 * self.target_count, dtype=np.float32),
            ]
        )
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.target_count)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, None]]:
        """Resets the environment to the initial state.

        Returns:
            Tuple[np.ndarray, Dict[str, None]]: The initial state of the environment and info dictionary.
        """
        self.step_count = 0
        self.current_location = 0
        self.visited_locations = []
        self.current_distances = self.distance_matrix[self.current_location]

        initial_state = np.concatenate(
            (
                np.array([self.current_location]),
                np.array(self.current_distances),
                np.array(self.coordinates).reshape(-1),
            ),
            dtype=np.float32,
        )
        return initial_state, {}

    def step(
        self, selected_action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        """Executes the action (move to the next location).

        Args:
            selected_action (int): The index of the next location to visit.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
                - New environment state
                - Reward for the chosen action
                - Termination flag
                - Truncation flag
                - Info dictionary
        """
        self.step_count += 1
        prev_location = self.current_location
        next_location = selected_action

        reward = self._calculate_reward(prev_location, next_location)
        self.visited_locations.append(next_location)

        next_distances = self.distance_matrix[next_location]
        is_terminated = bool(self.step_count == self.max_moves)
        is_truncated = False

        next_state = np.concatenate(
            [
                np.array([next_location]),
                next_distances,
                np.array(self.coordinates).reshape(-1),
            ],
            dtype=np.float32,
        )

        self.current_location, self.current_distances = next_location, next_distances
        return (next_state, reward, is_terminated, is_truncated, {})

    def _generate_random_targets(self, count: int) -> np.ndarray:
        """Generates random 2D coordinates for the target locations.

        Args:
            count (int): Number of target locations to generate.

        Returns:
            np.ndarray: Array of randomly generated 2D coordinates.
        """
        points = []
        while len(points) < count:
            x_coord = np.random.random() * self.grid_size
            y_coord = np.random.random() * self.grid_size
            if [x_coord, y_coord] not in points:
                points.append([x_coord, y_coord])

        return np.array(points)

    def _compute_distance_matrix(self, points: List) -> float:
        """Calculates the distance between all pairs of points.

        Args:
            points (List): List of 2D coordinates.

        Returns:
            np.ndarray: Distance matrix between all locations.
        """
        num_points = len(points)
        distances = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                distances[i, j] = np.linalg.norm(points[i] - points[j])
        return distances

    def _calculate_reward(self, previous_loc: int, next_loc: int) -> float:
        """Calculates the reward for moving from one location to another.

        Args:
            previous_loc (int): The current location of the agent.
            next_loc (int): The next location to move to.

        Returns:
            float: Reward based on the travel distance or penalty if revisiting a location.
        """
        if next_loc not in self.visited_locations:
            return -self.distance_matrix[previous_loc][next_loc]
        else:
            return -10000  # Penalty for revisiting


if __name__ == "__main__":
    target_count = 6
    episodes = 1000

    env = TSPEnvironment(target_count)

    # Initialize solvers
    dp_solver = create_dp_solver(env)
    mc_solver_first_visit = create_mc_solver(env, num_episodes=10000, method="first_visit")
    mc_solver_every_visit = create_mc_solver(env, num_episodes=10000, method="every_visit")
    mc_solver_epsilon_greedy = create_mc_epsilon_greedy_solver(
        env, num_episodes=episodes, method="first_visit", epsilon=0.1
    )

    solver_results = {
        "DP Solver": dp_solver,
        "MC First Visit": mc_solver_first_visit,
        "MC Every Visit": mc_solver_every_visit,
        "MC Epsilon Greedy": mc_solver_epsilon_greedy,
    }
    performance = {}

    for solver_name, solver in solver_results.items():
        print(f"\nEvaluating {solver_name}:")
        episode_rewards = []

        for episode in range(episodes):
            total_reward = 0
            observation, _ = env.reset()
            visited_states = []

            for _ in range(env.target_count):
                state = int(observation[0])
                visited_states.append(state)

                action = solver.get_action(state, visited_states)

                new_obs, reward, done, trunc, _ = env.step(action)
                total_reward += reward
                observation = new_obs

                if done or trunc:
                    break

            episode_rewards.append(total_reward)
            print(f"Episode {episode} : {total_reward} | Terminated : {done}")

        print(f"Average return over {episodes} episodes: {np.mean(episode_rewards)}")
        performance[solver_name] = np.mean(episode_rewards)

    print("\nSolver Performance Comparison:")
    for solver_name in solver_results:
        print(f"{solver_name} Average Return: {performance[solver_name]}")
