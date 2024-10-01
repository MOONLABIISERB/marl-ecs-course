"""Environment for Modified Travelling Salesman Problem."""

from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from numpy import typing as npt


class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits.

    The agent navigates a set of targets based on precomputed distances. It aims to visit
    all targets so maximize profits. The profits decay with time.
    """

    def __init__(
        self,
        num_targets: int = 10,
        max_area: int = 15,
        shuffle_time: int = 50001,
        seed: int = 42,
    ) -> None:
        """Initialize the TSP environment.

        Args:
            num_targets (int): No. of targets the agent needs to visit.
            max_area (int): Max. Square area where the targets are defined.
            shuffle_time (int): No. of episodes after which the profits ar to be shuffled.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()

        np.random.seed(seed)

        self.steps: int = 0
        self.episodes: int = 0

        self.shuffle_time: int = shuffle_time
        self.num_targets: int = num_targets

        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations: npt.NDArray[np.float32] = self._generate_points(self.num_targets)
        self.distances: npt.NDArray[np.float32] = self._calculate_distances(self.locations)

        # Initialize profits for each target
        self.initial_profits: npt.NDArray[np.float32] = np.arange(1, self.num_targets + 1, dtype=np.float32) * 10.0
        self.current_profits: npt.NDArray[np.float32] = self.initial_profits.copy()

        # Observation Space : {current loc (loc), target flag - visited or not, current profits, dist_array (distances), coordintates (locations)}
        self.obs_low = np.concatenate(
            [
                np.array([0], dtype=np.float32),  # Current location
                np.zeros(self.num_targets, dtype=np.float32),  # Check if targets were visited or not
                np.zeros(self.num_targets, dtype=np.float32),  # Array of all current profits values
                np.zeros(self.num_targets, dtype=np.float32),  # Distance to each target from current location
                np.zeros(2 * self.num_targets, dtype=np.float32),  # Cooridinates of all targets
            ]
        )

        self.obs_high = np.concatenate(
            [
                np.array([self.num_targets], dtype=np.float32),  # Current location
                np.ones(self.num_targets, dtype=np.float32),  # Check if targets were visited or not
                100 * np.ones(self.num_targets, dtype=np.float32),  # Array of all current profits values
                2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),  # Distance to each target from current location
                self.max_area * np.ones(2 * self.num_targets, dtype=np.float32),  # Cooridinates of all targets
            ]
        )

        # Action Space : {next_target}
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.num_targets)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, None]]:
        """Reset the environment to the initial state.

        Args:
            seed (Optional[int], optional): Seed to reset the environment. Defaults to None.
            options (Optional[dict], optional): Additional reset options. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, None]]: The initial state of the environment and an empty info dictionary.
        """
        self.steps: int = 0
        self.episodes += 1

        self.loc: int = 0
        self.visited_targets: npt.NDArray[np.float32] = np.zeros(self.num_targets)
        self.current_profits = self.initial_profits.copy()
        self.dist: List = self.distances[self.loc]

        if self.shuffle_time % self.episodes == 0:
            np.random.shuffle(self.initial_profits)

        state = np.concatenate(
            (
                np.array([self.loc]),
                self.visited_targets,
                self.initial_profits,
                np.array(self.dist),
                np.array(self.locations).reshape(-1),
            ),
            dtype=np.float32,
        )
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        """Take an action (move to the next target).

        Args:
            action (int): The index of the next target to move to.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
                - The new state of the environment.
                - The reward for the action.
                - A boolean indicating whether the episode has terminated.
                - A boolean indicating if the episode is truncated.
                - An empty info dictionary.
        """
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        self.current_profits -= self.distances[past_loc, next_loc]
        reward = self._get_rewards(next_loc)

        self.visited_targets[next_loc] = 1

        next_dist = self.distances[next_loc]
        terminated = bool(self.steps == self.max_steps)
        truncated = False

        next_state = np.concatenate(
            [
                np.array([next_loc]),
                self.visited_targets,
                self.current_profits,
                next_dist,
                np.array(self.locations).reshape(-1),
            ],
            dtype=np.float32,
        )

        self.loc, self.dist = next_loc, next_dist
        return (next_state, reward, terminated, truncated, {})

    def _generate_points(self, num_points: int) -> npt.NDArray[np.float32]:
        """Generate random 2D points representing target locations.

        Args:
            num_points (int): Number of points to generate.

        Returns:
            np.ndarray: Array of 2D coordinates for each target.
        """
        return np.random.uniform(low=0, high=self.max_area, size=(num_points, 2)).astype(np.float32)

    def _calculate_distances(self, locations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calculate the distance matrix between all target locations.

        Args:
            locations: List of 2D target locations.

        Returns:
            np.ndarray: Matrix of pairwise distances between targets.
        """
        n = len(locations)

        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, next_loc: int) -> float:
        """Calculate the reward based on the distance traveled, however if a target gets visited again then it incurs a high penalty.

        Args:
            next_loc (int): Next location of the agent.

        Returns:
            float: Reward based on the travel distance between past and next locations, or negative reward if repeats visit.
        """
        reward = self.current_profits[next_loc] if not self.visited_targets[next_loc] else -1e4
        return float(reward)


env = ModTSP(num_targets=10, max_area=15, shuffle_time=10)

def choose_action(state, Q, env, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state])  # Exploit

def SARSA(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """SARSA algorithm implementation."""
    # Initialize Q-table
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    episode_rewards = []
    episode_paths = []
    best_reward = -np.inf
    best_path = []

    for ep in range(num_episodes):
        # Reset
        state, _ = env.reset()
        state = tuple(state)

        action = choose_action(state, Q, env, epsilon)
        cumulative_reward = 0
        path = []

        for _ in range(env.max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_state)
            cumulative_reward += reward
            path.append(env.loc)

            next_action = choose_action(next_state, Q, env, epsilon)

            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state, action = next_state, next_action

            if terminated or truncated:
                break

        episode_rewards.append(cumulative_reward)
        episode_paths.append(path)

        if ep >= num_episodes - 10:
            print(f"Episode {ep+1}, Cumulative Reward: {cumulative_reward}, Path: {path}")

        if cumulative_reward > best_reward:
            best_reward = cumulative_reward
            best_path = path

    return Q, episode_rewards, episode_paths, best_reward, best_path

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_training(episode_rewards, num_plot_episodes, window_size=100):
    episode_rewards = episode_rewards[:num_plot_episodes]
    plt.figure(figsize=(20, 9))
    plt.plot(range(1, num_plot_episodes + 1), episode_rewards, label='Cumulative Reward')
    moving_avg = moving_average(episode_rewards, window_size)
    plt.plot(range(window_size, num_plot_episodes + 1), moving_avg, color='red', label=f'Moving Average (Window={window_size})')

    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title(f'SARSA Training: Episode vs. Cumulative Reward (First {num_plot_episodes} Episodes)')
    plt.legend()
    plt.show()

def plot_optimal_policy(env, best_path):
    plt.figure(figsize=(8, 8))
    
    for i, (x, y) in enumerate(env.locations):
        plt.scatter(x, y, s=100)
        plt.text(x, y, f"{i}", fontsize=12, ha='right', va='bottom', color='blue')

    for i in range(len(best_path) - 1):
        x1, y1 = env.locations[best_path[i]]
        x2, y2 = env.locations[best_path[i+1]]
        plt.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, length_includes_head=True, color='green')

    x1, y1 = env.locations[best_path[-1]]
    x2, y2 = env.locations[best_path[0]]
    plt.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, length_includes_head=True, color='red', linestyle='--')

    plt.title("Optimal Policy for TSP")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True)
    plt.show()

def main(num_plot_episodes):
    num_targets = 10
    env = ModTSP(num_targets)

    # Hyperparameters
    num_episodes = 40000
    alpha = 0.00001
    gamma = 0.99
    epsilon = 0.0001

    Q, episode_rewards, episode_paths, best_reward, best_path = SARSA(env, num_episodes, alpha, gamma, epsilon)
    plot_training(episode_rewards, num_plot_episodes)

    # Print the best reward and path
    print(f"Best Reward: {best_reward}")
    print(f"Best Path: {best_path}")
    plot_optimal_policy(env, best_path)

if __name__ == "__main__":
    num_plot_episodes = 40000
    main(num_plot_episodes)
