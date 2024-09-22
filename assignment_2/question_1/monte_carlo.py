import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple

class TSP(gym.Env):
    """Traveling Salesman Problem (TSP) RL environment for persistent monitoring."""

    def __init__(self, num_targets: int, max_area: int = 30, seed: int = None) -> None:
        super().__init__()
        if seed is not None:
            np.random.seed(seed=seed)

        self.num_targets: int = num_targets
        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations: np.ndarray = self._generate_points(self.num_targets)
        self.distances: np.ndarray = self._calculate_distances(self.locations)

        self.obs_low = np.concatenate(
            [
                np.array([0], dtype=np.float32),
                np.zeros(self.num_targets, dtype=np.float32),
                np.zeros(2 * self.num_targets, dtype=np.float32),
            ]
        )

        self.obs_high = np.concatenate(
            [
                np.array([self.num_targets], dtype=np.float32),
                2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),
                self.max_area * np.ones(2 * self.num_targets, dtype=np.float32),
            ]
        )

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.num_targets)

        # Initialize attributes
        self.steps: int = 0
        self.loc: int = 0
        self.visited_targets: List = []
        self.dist: List = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, None]]:
        self.steps: int = 0
        self.loc: int = 0
        self.visited_targets: List = []
        self.dist: List = self.distances[self.loc].tolist()

        state = np.concatenate(
            (
                np.array([self.loc]),
                np.array(self.dist),
                np.array(self.locations).reshape(-1),
            ),
            dtype=np.float32,
        )
        return state, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        reward = self._get_rewards(past_loc, next_loc)
        self.visited_targets.append(next_loc)

        next_dist = self.distances[next_loc].tolist()
        terminated = bool(self.steps == self.max_steps)
        truncated = False

        next_state = np.concatenate(
            [
                np.array([next_loc]),
                np.array(next_dist),
                np.array(self.locations).reshape(-1),
            ],
            dtype=np.float32,
        )

        self.loc, self.dist = next_loc, next_dist
        return (next_state, reward, terminated, truncated, {})

    def _generate_points(self, num_points: int) -> np.ndarray:
        points = []
        while len(points) < num_points:
            x = np.random.random() * self.max_area
            y = np.random.random() * self.max_area
            if [x, y] not in points:
                points.append([x, y])
        return np.array(points)

    def _calculate_distances(self, locations: List) -> np.ndarray:
        n = len(locations)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, past_loc: int, next_loc: int) -> float:
        if next_loc not in self.visited_targets:
            reward = -self.distances[past_loc][next_loc]
        else:
            reward = -10000
        return reward

def monte_carlo_first_visit(env, num_episodes=10000, gamma=0.99, epsilon=0.1):
    pass
    # Q = {}
    # N = {}
    # policy = {}

    # def get_action(state):
    #     state_key = tuple(state)
    #     if state_key not in policy or np.random.random() < epsilon:
    #         return env.action_space.sample()
    #     return policy[state_key]

    # for episode in range(num_episodes):
    #     state, _ = env.reset()
    #     episode_data = []
    #     for t in range(env.max_steps):
    #         action = get_action(state)
    #         next_state, reward, terminated, truncated, _ = env.step(action)
    #         episode_data.append((state, action, reward))
    #         state = next_state
    #         if terminated or truncated:
    #             break

    #     G = 0
    #     visited_state_actions = set()
    #     for t in range(len(episode_data) - 1, -1, -1):
    #         state, action, reward = episode_data[t]
    #         G = gamma * G + reward

    #         state_action = (tuple(state), action)
    #         if state_action not in visited_state_actions:  # First-visit
    #             if state_action not in N:
    #                 N[state_action] = 0
    #                 Q[state_action] = 0
    #             N[state_action] += 1
    #             Q[state_action] += (G - Q[state_action]) / N[state_action]
    #             visited_state_actions.add(state_action)

    #         # Update policy
    #         state_key = tuple(state)
    #         if state_key not in policy or Q[state_key, action] > Q.get((state_key, policy[state_key]), float('-inf')):
    #             policy[state_key] = action

    #     if episode % 100 == 0:
    #         print(f"First-visit MC: Episode {episode} completed")

    # return policy



def monte_carlo_every_visit(env, num_episodes=10000, gamma=0.99, epsilon=0.1):
    pass
    # Q = {}
    # N = {}
    # policy = {}

    # def get_action(state):
    #     state_key = tuple(state)
    #     if state_key not in policy or np.random.random() < epsilon:
    #         return env.action_space.sample()
    #     return policy[state_key]

    # for episode in range(num_episodes):
    #     state, _ = env.reset()
    #     episode_data = []
    #     for t in range(env.max_steps):
    #         action = get_action(state)
    #         next_state, reward, terminated, truncated, _ = env.step(action)
    #         episode_data.append((state, action, reward))
    #         state = next_state
    #         if terminated or truncated:
    #             break

    #     G = 0
    #     for t in range(len(episode_data) - 1, -1, -1):
    #         state, action, reward = episode_data[t]
    #         G = gamma * G + reward

    #         state_action = (tuple(state), action)
    #         if state_action not in N:
    #             N[state_action] = 0
    #             Q[state_action] = 0
    #         N[state_action] += 1
    #         Q[state_action] += (G - Q[state_action]) / N[state_action]

    #         # Update policy
    #         state_key = tuple(state)
    #         if state_key not in policy or Q[state_key, action] > Q.get((state_key, policy[state_key]), float('-inf')):
    #             policy[state_key] = action

    #     if episode % 100 == 0:
    #         print(f"Every-visit MC: Episode {episode} completed")

    # return policy



if __name__ == "__main__":
    num_targets = 50 
    env = TSP(num_targets)
    
    print("Starting First-Visit Monte Carlo method...")
    first_visit_policy = monte_carlo_first_visit(env)
    
    print("Starting Every-Visit Monte Carlo method...")
    every_visit_policy = monte_carlo_every_visit(env)
    
    print("Evaluating the policies...")
    
    # Evaluate First-Visit policy
    total_reward_first = 0
    state, _ = env.reset()
    for _ in range(num_targets):
        action = first_visit_policy.get(tuple(state), env.action_space.sample())
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward_first += reward
        if terminated or truncated:
            break
    
    # Evaluate Every-Visit policy
    total_reward_every = 0
    state, _ = env.reset()
    for _ in range(num_targets):
        action = every_visit_policy.get(tuple(state), env.action_space.sample())
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward_every += reward
        if terminated or truncated:
            break
    
    print(f"Total reward using First-Visit Monte Carlo: {total_reward_first}")
    print(f"Total reward using Every-Visit Monte Carlo: {total_reward_every}")
    print(f"Number of states in First-Visit policy: {len(first_visit_policy)}")
    print(f"Number of states in Every-Visit policy: {len(every_visit_policy)}")
