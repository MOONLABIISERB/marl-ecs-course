import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from itertools import product
import gymnasium as gym
import numpy as np
from numpy import typing as npt
import random


class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits."""

    def __init__(
        self,
        num_targets: int = 10,
        max_area: int = 15,
        shuffle_time: int = 10,
        seed: int = 42,
    ) -> None:
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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, None]]:
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
        return np.random.uniform(low=0, high=self.max_area, size=(num_points, 2)).astype(np.float32)

    def _calculate_distances(self, locations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        n = len(locations)
        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, next_loc: int) -> float:
        reward = self.current_profits[next_loc] if not self.visited_targets[next_loc] else -1e4
        return float(reward)


def load_q_table(file_path):
    """Loads the pre-trained Q- file from train.py"""
    return np.load(file_path, allow_pickle=True).item()


def best_action(Q, a, num_targets):
    """Selects the best action based on the Q-table."""
    max_value = -float('inf')
    best_action = None
    for b in range(num_targets):
        if (tuple(a), b) in Q:
            current_value = Q[(tuple(a), b)]
            if current_value > max_value:
                max_value = current_value
                best_action = b
    return best_action


def main():
    num_targets = 10
    env = ModTSP(num_targets)
    Q = load_q_table("q_table.npy")  # Load the trained Q-table
    max_reward = -float('inf')
    ep_rets = []
    for ep in range(100):
        ret = 0
        obs, _ = env.reset()
        state = obs[:11].astype(int)
        state = [int(x) for x in state]
        visited_path = []
        visited_path_loc = []
        for _ in range(100):
            if visited_path == []:  # first state will always be random
                action = env.action_space.sample()
            else:  # following policy extracted from Q table
                action = best_action(Q, state, num_targets)  # Use the trained Q-table to select the action

            obs_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = obs_[:11].astype(int)
            next_state = [int(x) for x in next_state]
            ret += reward
            visited_path.append(next_state[0])
            visited_path_loc.append(env.locations[next_state[0]])
            if done:
                break

            state = next_state

        ep_rets.append(ret)
        if ret > max_reward:  # finding path with max rewards/profits
            max_reward = ret
            best_path = visited_path_loc
        print(f"Test Episode {ep}: Total Reward: {ret}   Visited Path: {visited_path}")

    print(f"Average Reward over 100 test episodes: {np.mean(ep_rets)}")

    best_path = np.array(best_path)
    plt.figure(figsize=(10, 8))
    plt.plot(best_path[:, 0], best_path[:, 1], marker='o', linestyle='-', color='b', label='Path Taken')

    # Labeling each target location
    for i, (x, y) in enumerate(env.locations):
        plt.annotate(f'Target {i}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    plt.title('Best Path Taken Based on Maximum Reward')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
