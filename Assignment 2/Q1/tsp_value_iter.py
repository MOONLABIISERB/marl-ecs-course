from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np


class TSP(gym.Env):
    """Traveling Salesman Problem (TSP) RL environment for persistent monitoring."""

    def __init__(self, num_targets: int, max_area: int = 30, seed: int = None) -> None:
        super().__init__()
        if seed is not None:
            np.random.seed(seed=seed)

        self.steps: int = 0
        self.num_targets: int = num_targets
        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations: np.ndarray = self._generate_points(self.num_targets)
        self.distances: np.ndarray = self._calculate_distances(self.locations)

        # Observation and Action Space
        self.obs_low = np.concatenate(
            [np.array([0], dtype=np.float32),
             np.zeros(self.num_targets, dtype=np.float32),
             np.zeros(2 * self.num_targets, dtype=np.float32)]
        )
        self.obs_high = np.concatenate(
            [np.array([self.num_targets], dtype=np.float32),
             2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),
             self.max_area * np.ones(2 * self.num_targets, dtype=np.float32)]
        )

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.num_targets)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, init_loc: Optional[int] = 0) -> Tuple[np.ndarray, Dict[str, None]]:
        self.steps = 0
        self.loc = init_loc
        self.visited_targets = []
        self.dist = self.distances[self.loc]

        state = np.concatenate(
            (np.array([self.loc]),
             np.array(self.dist),
             np.array(self.locations).reshape(-1)),
            dtype=np.float32,
        )
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        reward = self._get_rewards(past_loc, next_loc)
        self.visited_targets.append(next_loc)

        next_dist = self.distances[next_loc]
        terminated = self.steps == self.max_steps
        truncated = False

        next_state = np.concatenate(
            [np.array([next_loc]),
             next_dist,
             np.array(self.locations).reshape(-1)],
            dtype=np.float32,
        )

        self.loc, self.dist = next_loc, next_dist
        return next_state, reward, terminated, truncated, {}

    def _generate_points(self, num_points: int) -> np.ndarray:
        points = []
        while len(points) < num_points:
            x = np.random.random() * self.max_area
            y = np.random.random() * self.max_area
            if [x, y] not in points:
                points.append([x, y])
        return np.array(points)

    def _calculate_distances(self, locations: List) -> float:
        n = len(locations)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, past_loc: int, next_loc: int) -> float:
        if next_loc not in self.visited_targets:
            return -self.distances[past_loc][next_loc]
        return -10000


if __name__ == "__main__":
    num_targets = 6
    max_episodes = 10000
    max_steps = 10

    env = TSP(num_targets)
    
    policy = {}
    gamma = 0.9  # Discount factor
    theta = 1e-6  # Convergence threshold
    
    # Initialize Value function
    V = np.zeros((env.num_targets,))
    
    # Value Iteration
    for ep in range(max_episodes):
        delta = 0
        for s in range(env.num_targets):
            v = V[s]
            # Bellman update
            new_value = max(
                [-env.distances[s][a] + gamma * V[a] for a in range(env.num_targets) if a != s]
            )
            V[s] = new_value
            delta = max(delta, abs(v - new_value))
        
        if delta < theta:
            print(f"Value Iteration converged after {ep + 1} episodes.")
            break
    
    # Derive policy from value function
    for s in range(env.num_targets):
        best_action = np.argmax(
            [-env.distances[s][a] + gamma * V[a] for a in range(env.num_targets) if a != s]
        )
        policy[s] = best_action

    print("Optimal Policy:", policy)
    
    # Test the learned policy
    obs, _ = env.reset(init_loc=0)
    print("\nTesting learned policy:")
    
    visited = set()
    terminated = False

    while len(visited) < num_targets:
        action = policy[env.loc]

        # If the action has already been visited, find a new action
        if action in visited:
            # Find the first unvisited action
            unvisited_actions = [a for a in range(env.num_targets) if a not in visited and a != env.loc]
            if unvisited_actions:
                action = unvisited_actions[0]  # Choose the first unvisited action
            else:
                print("All actions visited, stopping.")
                break

        obs_, reward, terminated, truncated, info = env.step(action)
        visited.add(action)
        
        print(f"Taken action {action}, Reward: {reward}")

        if terminated:
            break