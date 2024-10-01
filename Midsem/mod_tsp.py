import numpy as np
from numpy import typing as npt
import gymnasium as gym
import matplotlib.pyplot as plt  # Import for plotting
from typing import Dict, List, Optional, Tuple


class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits.

    The agent navigates a set of targets based on precomputed distances. It aims to visit
    all targets to maximize profits. The profits decay with time.
    """

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

        # Initialize profits for each target
        self.initial_profits: npt.NDArray[np.float32] = np.arange(1, self.num_targets + 1, dtype=np.float32) * 10.0
        self.current_profits: npt.NDArray[np.float32] = self.initial_profits.copy()

        # Observation Space
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

    def reset(self) -> Tuple[np.ndarray, Dict[str, None]]:
        """Reset the environment to the initial state."""
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
        return next_state, reward, terminated, truncated, {}

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
        """Calculate the reward based on the distance traveled."""
        reward = self.current_profits[next_loc] if not self.visited_targets[next_loc] else -1e4
        return float(reward)


class QLearner:
    """Q-learning agent for the Modified TSP environment."""

    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9):
        """Initialize Q-learning agent."""
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_matrix = {}
        self.temporal_difference_errors = []

    def select_action(self, current_state):
        """Select the best action using a greedy policy based on the current state's Q-values."""
        state_key = str(current_state)
        if state_key not in self.q_matrix:
            self.q_matrix[state_key] = np.zeros(self.num_actions)
        return np.argmax(self.q_matrix[state_key])

    def update_q_matrix(self, current_state, selected_action, reward, future_state):
        """Update the Q-value for the given state-action pair."""
        state_key = str(current_state)
        next_state_key = str(future_state)

        if next_state_key not in self.q_matrix:
            self.q_matrix[next_state_key] = np.zeros(self.num_actions)

        best_future_action = np.argmax(self.q_matrix[next_state_key])
        q_target = reward + self.discount_factor * self.q_matrix[next_state_key][best_future_action]
        td_error = q_target - self.q_matrix[state_key][selected_action]

        self.temporal_difference_errors.append(abs(td_error))
        self.q_matrix[state_key][selected_action] += self.learning_rate * td_error


def main() -> None:
    """Main function to run Q-learning agent in Modified TSP environment."""
    num_targets = 10
    shuffle_time = 10
    num_episodes = 10000

    env = ModTSP(num_targets, shuffle_time=shuffle_time)
    agent = QLearner(num_actions=num_targets)

    ep_rets = []  # Track cumulative rewards
    avg_losses = []  # Track average loss (TD error)

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update_q_matrix(state, action, reward, next_state)
            total_reward += reward
            state = next_state

        ep_rets.append(total_reward)
        avg_losses.append(np.mean(agent.temporal_difference_errors))

        print(f"Episode {ep} / {num_episodes}: Total Reward Collected: {total_reward}")

    # Save the Q-matrix (learned Q-values) to a file after training
    np.save('q_matrix.npy', agent.q_matrix)  # Save Q-matrix as a .npy file

    # Plot cumulative rewards and average losses
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # First plot: Cumulative reward per episode
    axes[0].plot(ep_rets, label='Cumulative Reward', alpha=0.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_title('Cumulative Reward per Episode')
    axes[0].legend()

    # Second plot: Average TD error (loss) per episode
    axes[1].plot(avg_losses, label='Average TD Error (Loss)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Average Loss')
    axes[1].set_title('Average Loss per Episode')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
