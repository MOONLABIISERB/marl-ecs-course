import numpy as np
import random
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
# from exp import q_value

class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits."""

    def __init__(self, num_targets: int = 10, max_area: int = 15, shuffle_time: int = 10, seed: int = 42) -> None:
        super().__init__()
        np.random.seed(seed)
        self.steps: int = 0
        self.episodes: int = 0
        self.shuffle_time: int = shuffle_time
        self.num_targets: int = num_targets
        self.max_steps: int = num_targets
        self.max_area: int = max_area
        self.locations = self._generate_points(self.num_targets)
        self.distances = self._calculate_distances(self.locations)
        self.initial_profits = np.arange(1, self.num_targets + 1, dtype=np.float32) * 10.0
        self.current_profits = self.initial_profits.copy()

        self.obs_low = np.concatenate([
            np.array([0], dtype=np.float32),
            np.zeros(self.num_targets, dtype=np.float32),
            np.zeros(self.num_targets, dtype=np.float32),
            np.zeros(2 * self.num_targets, dtype=np.float32),
        ])
        self.obs_high = np.concatenate([
            np.array([self.num_targets], dtype=np.float32),
            100 * np.ones(self.num_targets, dtype=np.float32),
            2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),
            self.max_area * np.ones(2 * self.num_targets, dtype=np.float32),
        ])

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.num_targets)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, None]]:
        self.steps = 0
        self.episodes += 1
        self.loc = 0
        self.visited_targets = [self.loc]  # Start at the first location and mark it as visited
        self.dist = self.distances[self.loc]
        if self.episodes % self.shuffle_time == 0:
            np.random.shuffle(self.initial_profits)
        state = np.concatenate(
            (np.array([self.loc]), np.array(self.dist), np.array(self.locations).reshape(-1)),
            dtype=np.float32,
        )
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        if next_loc in self.visited_targets:  # Penalize for revisiting a location
            reward = -1e4
        else:
            reward = self.current_profits[next_loc] - self.distances[past_loc, next_loc]
            self.visited_targets.append(next_loc)

        self.current_profits -= self.distances[past_loc, next_loc]
        next_dist = self.distances[next_loc]

        terminated = len(self.visited_targets) == self.num_targets  # Episode ends when all locations are visited
        truncated = False

        next_state = np.concatenate(
            [np.array([next_loc]), next_dist, np.array(self.locations).reshape(-1)],
            dtype=np.float32,
        )

        self.loc, self.dist = next_loc, next_dist
        return next_state, reward, terminated, truncated, {}

    def _generate_points(self, num_points: int) -> np.ndarray:
        return np.random.uniform(low=0, high=self.max_area, size=(num_points, 2)).astype(np.float32)

    def _calculate_distances(self, locations: np.ndarray) -> np.ndarray:
        n = len(locations)
        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances


def epsilon_greedy_policy(Q, state, epsilon, action_space):
    """Choose an action using an epsilon-greedy strategy."""
    if np.random.rand() < epsilon:
        return action_space.sample()  # Explore
    else:
        return np.argmax(Q[state])  # Exploit


def main() -> None:
    num_targets = 10
    env = ModTSP(num_targets)
    
    distances = env._calculate_distances(env._generate_points(10))
    num_episodes = 1000
    max_steps_per_episode = num_targets  # Ensure that we stop when all locations are visited
    alpha = 0.01  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 0.1  # Epsilon for epsilon-greedy policy

    # Initialize Q-table
    Q = np.zeros((num_targets, num_targets))  # Q[state, action]

    # Track the best path and maximum reward
    best_path = []
    best_actions = []
    best_total_reward = -np.inf

    for ep in range(num_episodes):
        # Reset the environment
        state, _ = env.reset()
        state = int(state[0])  # Current location as the state
        action = epsilon_greedy_policy(Q, state, epsilon, env.action_space)

        total_reward = 0
        path = [state]  # Track path (sequence of states visited)
        actions_taken = [action]  # Track actions taken

        distance_traveled = 0
        
        for step in range(max_steps_per_episode):
            # Take action and observe the next state, reward, and termination status
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state[0])  # Next location as the next state
            
            if next_state not in path:
                step_reward = reward - distance_traveled # modified reward
            else: 
                step_reward = - 1e4 # reward for repeating step
            # total_reward += reward

            # Choose next action using epsilon-greedy policy
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.action_space)

            # SARSA update rule
            Q[state, action] = Q[state, action] + alpha * (
                step_reward + gamma * Q[next_state, next_action] - Q[state, action]
            )
            distance_traveled += distances[state][action]            
            state, action = next_state, next_action
            path.append(state)
            actions_taken.append(action)
            
            

            if terminated or truncated:
                break

    

        print(f"Episode {ep + 1}/{num_episodes} - Total Reward: {Q}")
        print(f"Path: {path}")


if __name__ == "__main__":
    main()
