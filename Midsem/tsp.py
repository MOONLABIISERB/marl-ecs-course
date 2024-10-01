import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from numpy import typing as npt
from sarsa import SARSA


class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits."""
    def __init__(
        self,
        num_targets: int = 10,
        max_area: int = 15,
        shuffle_time: int = 10,
        seed: int = 42,
    ) -> None:
        """Initialize the TSP environment."""
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
        self.initial_profits: npt.NDArray[np.float32] = np.arange(1, self.num_targets + 1, dtype=np.float32) * 10.0
        self.current_profits: npt.NDArray[np.float32] = self.initial_profits.copy()

        self.obs_low = np.concatenate(
            [
                np.array([0], dtype=np.float32),
                np.zeros(self.num_targets, dtype=np.float32),
                np.zeros(self.num_targets, dtype=np.float32),
                np.zeros(self.num_targets, dtype=np.float32),
                np.zeros(2 * self.num_targets, dtype=np.float32),
            ]
        )

        self.obs_high = np.concatenate(
            [
                np.array([self.num_targets], dtype=np.float32),
                np.ones(self.num_targets, dtype=np.float32),
                100 * np.ones(self.num_targets, dtype=np.float32),
                2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),
                self.max_area * np.ones(2 * self.num_targets, dtype=np.float32),
            ]
        )

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.num_targets)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, None]]:
        """Reset the environment."""
        self.steps = 0
        self.episodes += 1
        self.loc = 0
        self.visited_targets = np.zeros(self.num_targets)
        self.current_profits = self.initial_profits.copy()
        self.dist = self.distances[self.loc]

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
        """Take an action (move to the next target)."""
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
        """Generate random 2D points representing target locations."""
        return np.random.uniform(low=0, high=self.max_area, size=(num_points, 2)).astype(np.float32)

    def _calculate_distances(self, locations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calculate the distance matrix between all target locations."""
        n = len(locations)
        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, next_loc: int) -> float:
        """Calculate the reward."""
        reward = self.current_profits[next_loc] if not self.visited_targets[next_loc] else -1e4
        return float(reward)


def graph(csv_file):
    episodes = []
    cumulative_rewards = []
    episodic_rewards = []

    # Read data from CSV file
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            episodes.append(int(row[0]))
            episodic_rewards.append(float(row[1]))
            cumulative_rewards.append(float(row[2]))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episodic_rewards, label='Episodic Reward', color='g', marker='o', linestyle='--', alpha=0.5)
    plt.plot(episodes, cumulative_rewards, label='Cumulative Reward', color='b', marker='x')
    plt.title('Episode vs Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()
    plt.show()


def main() -> None:
    """Main function."""
    num_targets = 10
    cumulative_rewards = []
    env = ModTSP(num_targets)
    obs = env.reset()
    ep_rets = []
    distances = env._calculate_distances(env._generate_points(10))
    states = np.array(list(range(num_targets)), dtype=int)
    actions = states
    rewards = 100 * np.ones(num_targets, dtype=np.float32)
    sarsa = SARSA(states, actions, rewards, distances)

    # Open a CSV file to store the results
    with open('rewards.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Episodic Reward', 'Cumulative Reward'])

        for ep in range(150):
            ret = 0
            obs, _ = env.reset()
            state = int(obs[0])
            Q, cum, epi = sarsa.training(state)

            for _ in range(150):
                action = np.argmax(Q[state])
                obs_, reward, terminated, truncated, info = env.step(action)
                ret += reward
                state = int(obs_[0])
                if terminated or truncated:
                    break

            ep_rets.append(ret)
            mean_rewards = np.mean(ep_rets)
            print(f"Episode {ep}: {ret}, Mean Reward: {mean_rewards}")
            cumulative_rewards.append(mean_rewards)

            # Save rewards to CSV
            writer.writerow([ep + 1, ret, mean_rewards])

    # Plot graph from CSV file
    graph('rewards.csv')


if __name__ == "__main__":
    main()
