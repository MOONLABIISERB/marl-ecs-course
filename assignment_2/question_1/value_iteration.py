# import numpy as np
# import gymnasium as gym
# from typing import Dict, List, Optional, Tuple

# class TSP(gym.Env):
#     """Traveling Salesman Problem (TSP) RL environment for persistent monitoring."""

#     def __init__(self, num_targets: int, max_area: int = 30, seed: int = None) -> None:
#         super().__init__()
#         if seed is not None:
#             np.random.seed(seed=seed)

#         self.num_targets: int = num_targets
#         self.max_steps: int = num_targets
#         self.max_area: int = max_area

#         self.locations: np.ndarray = self._generate_points(self.num_targets)
#         self.distances: np.ndarray = self._calculate_distances(self.locations)

#         self.obs_low = np.concatenate(
#             [
#                 np.array([0], dtype=np.float32),
#                 np.zeros(self.num_targets, dtype=np.float32),
#                 np.zeros(2 * self.num_targets, dtype=np.float32),
#             ]
#         )

#         self.obs_high = np.concatenate(
#             [
#                 np.array([self.num_targets], dtype=np.float32),
#                 2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),
#                 self.max_area * np.ones(2 * self.num_targets, dtype=np.float32),
#             ]
#         )

#         self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
#         self.action_space = gym.spaces.Discrete(self.num_targets)

#         # Initialize attributes
#         self.steps: int = 0
#         self.loc: int = 0
#         self.visited_targets: List = []
#         self.dist: List = []

#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         options: Optional[dict] = None,
#     ) -> Tuple[np.ndarray, Dict[str, None]]:
#         self.steps: int = 0
#         self.loc: int = 0
#         self.visited_targets: List = []
#         self.dist: List = self.distances[self.loc].tolist()

#         state = np.concatenate(
#             (
#                 np.array([self.loc]),
#                 np.array(self.dist),
#                 np.array(self.locations).reshape(-1),
#             ),
#             dtype=np.float32,
#         )
#         return state, {}

#     def step(
#         self, action: int
#     ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
#         self.steps += 1
#         past_loc = self.loc
#         next_loc = action

#         reward = self._get_rewards(past_loc, next_loc)
#         self.visited_targets.append(next_loc)

#         next_dist = self.distances[next_loc].tolist()
#         terminated = bool(self.steps == self.max_steps)
#         truncated = False

#         next_state = np.concatenate(
#             [
#                 np.array([next_loc]),
#                 np.array(next_dist),
#                 np.array(self.locations).reshape(-1),
#             ],
#             dtype=np.float32,
#         )

#         self.loc, self.dist = next_loc, next_dist
#         return (next_state, reward, terminated, truncated, {})

#     def _generate_points(self, num_points: int) -> np.ndarray:
#         points = []
#         while len(points) < num_points:
#             x = np.random.random() * self.max_area
#             y = np.random.random() * self.max_area
#             if [x, y] not in points:
#                 points.append([x, y])
#         return np.array(points)

#     def _calculate_distances(self, locations: List) -> np.ndarray:
#         n = len(locations)
#         distances = np.zeros((n, n))
#         for i in range(n):
#             for j in range(n):
#                 distances[i, j] = np.linalg.norm(locations[i] - locations[j])
#         return distances

#     def _get_rewards(self, past_loc: int, next_loc: int) -> float:
#         if next_loc not in self.visited_targets:
#             reward = -self.distances[past_loc][next_loc]
#         else:
#             reward = -10000
#         return reward


# def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=100):
#     num_states = env.observation_space.shape[0]
#     num_actions = env.action_space.n
    
#     V = np.zeros(num_states)
#     for i in range(max_iterations):
#         delta = 0
#         for s in range(num_states):
#             v = V[s]
#             Q = np.zeros(num_actions)
#             for a in range(num_actions):
#                 next_state, reward, terminated, truncated, _ = env.step(a)
#                 Q[a] = reward + gamma * V[int(next_state[0])]  # Use the location as the state index
#                 env.reset()  # Reset the environment after each action
#             V[s] = np.max(Q)
#             delta = max(delta, abs(v - V[s]))
#         if delta < theta:
#             break
    
#     # Extract policy
#     policy = np.zeros(num_states, dtype=int)
#     for s in range(num_states):
#         Q = np.zeros(num_actions)
#         for a in range(num_actions):
#             next_state, reward, terminated, truncated, _ = env.step(a)
#             Q[a] = reward + gamma * V[int(next_state[0])]
#             env.reset()
#         policy[s] = np.argmax(Q)
    
#     return V, policy



# if __name__ == "__main__":
#     num_targets = 50 
#     env = TSP(num_targets)
    
#     print("Starting Value Iteration...")
#     optimal_value, optimal_policy = value_iteration(env)
    
#     print("Evaluating the policy...")
#     total_reward = 0
#     state, _ = env.reset()
#     for _ in range(num_targets):
#         action = optimal_policy[int(state[0])]
#         state, reward, terminated, truncated, _ = env.step(action)
#         total_reward += reward
#         if terminated or truncated:
#             break

#     print(f"Total reward using Value Iteration: {total_reward}")
#     print(f"Optimal policy: {optimal_policy}")

"""Environment for Travelling Salesman Problem."""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

class TSP(gym.Env):
    """Traveling Salesman Problem (TSP) RL environment for persistent monitoring.

    The agent navigates a set of targets based on precomputed distances. It aims to visit
    all targets in the least number of steps, with rewards determined by the distance traveled.
    """

    def __init__(self, num_targets: int, max_area: int = 30, seed: int = None) -> None:
        """Initialize the TSP environment.

        Args:
            num_targets (int): Number of targets the agent needs to visit.
            max_area (int): Max Square area where the targets are defined. Defaults to 30
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed=seed)

        self.steps: int = 0
        self.num_targets: int = num_targets

        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations: np.ndarray = self._generate_points(self.num_targets)
        self.distances: np.ndarray = self._calculate_distances(self.locations)

        # Observation Space : {current loc (loc), dist_array (distances), coordintates (locations)}
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

        self.loc: int = 0
        self.visited_targets: List = []
        self.dist: List = self.distances[self.loc]

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

        reward = self._get_rewards(past_loc, next_loc)
        self.visited_targets.append(next_loc)

        next_dist = self.distances[next_loc]
        terminated = bool(self.steps == self.max_steps)
        truncated = False

        next_state = np.concatenate(
            [
                np.array([next_loc]),
                next_dist,
                np.array(self.locations).reshape(-1),
            ],
            dtype=np.float32,
        )

        self.loc, self.dist = next_loc, next_dist
        return (next_state, reward, terminated, truncated, {})

    def _generate_points(self, num_points: int) -> np.ndarray:
        """Generate random 2D points representing target locations.

        Args:
            num_points (int): Number of points to generate.

        Returns:
            np.ndarray: Array of 2D coordinates for each target.
        """
        points = []
        # Generate n random 2D points within the 10x10 grid
        while len(points) < num_points:
            x = np.random.random() * self.max_area
            y = np.random.random() * self.max_area
            if [x, y] not in points:
                points.append([x, y])

        return np.array(points)



    def _calculate_distances(self, locations: List) -> float:
        """Calculate the distance matrix between all target locations.

        Args:
            locations (List): List of 2D target locations.

        Returns:
            np.ndarray: Matrix of pairwise distances between targets.
        """
        n = len(locations)

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, past_loc: int, next_loc: int) -> float:
        """Calculate the reward based on the distance traveled, however if a target gets visited again then it incurs a high penalty.

        Args:
            past_loc (int): Previous location of the agent.
            next_loc (int): Next location of the agent.

        Returns:
            float: Reward based on the travel distance between past and next locations, or negative reward if repeats visit.
        """
        if next_loc not in self.visited_targets:
            reward = -self.distances[past_loc][next_loc]
        else:
            reward = -10000
        return reward


    def value_iteration(self, gamma=0.9, theta=1e-6, max_iterations=1000):
        pass
        # """Perform value iteration to solve the TSP."""
        # V = np.zeros((self.num_targets,))  # Initialize value function
        # policy = np.zeros((self.num_targets,), dtype=int)  # Initialize policy
        
        # for iteration in range(max_iterations):
        #     delta = 0  # Measure of improvement
        #     for state in range(self.num_targets):
        #         v = V[state]  # Current value of the state

        #         # For each action, calculate the expected return
        #         action_values = np.zeros((self.num_targets,))
        #         for action in range(self.num_targets):
        #             if action == state or action in self.visited_targets:
        #                 continue
        #             next_state = action
        #             reward = self._get_rewards(state, next_state)
        #             action_values[action] = reward + gamma * V[next_state]

        #         # Update the value function
        #         V[state] = np.max(action_values)
        #         delta = max(delta, abs(v - V[state]))

        #     # Convergence check
        #     if delta < theta:
        #         break

        # # Derive policy from the value function
        # for state in range(self.num_targets):
        #     action_values = np.zeros((self.num_targets,))
        #     for action in range(self.num_targets):
        #         if action == state or action in self.visited_targets:
        #             continue
        #         next_state = action
        #         reward = self._get_rewards(state, next_state)
        #         action_values[action] = reward + gamma * V[next_state]

        #     policy[state] = np.argmax(action_values)

        # return V, policy




if __name__ == "__main__":
    num_targets = 10

    env = TSP(num_targets)
    V, policy = env.value_iteration()
    print("Optimal Value Function:", V)
    print("Optimal Policy:", policy)

    ep_rets = []

    for ep in range(100):
        ret = 0
        obs, _ = env.reset()  # Reset the environment to get the initial state
        for _ in range(100):
            action = policy[int(obs[0])]  # Use the learned policy to select action
            
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += reward

            if done:
                break

        ep_rets.append(ret)
        print(f"Episode {ep + 1}: Total Reward = {ret}")

    print("Average Reward over 100 Episodes:", np.mean(ep_rets))
