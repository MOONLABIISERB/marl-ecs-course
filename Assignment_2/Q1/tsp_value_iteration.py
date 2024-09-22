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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        initial_state: Optional[int] = None,
        visited_targets: Optional[List[int]] = None 
    ) -> Tuple[np.ndarray, Dict[str, None]]:
        """Reset the environment to the initial state.

        Args:
            seed (Optional[int], optional): Seed to reset the environment. Defaults to None.
            options (Optional[dict], optional): Additional reset options. Defaults to None.
            initial_state (Optional[int], optional): Custom initial state to reset the environment to. Defaults to None.
            visited_targets (Optional[List[int]], optional): List of visited targets to reset the environment with. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, None]]: The initial state of the environment and an empty info dictionary.
        """
        self.steps: int = 0

        self.loc: int = initial_state if initial_state is not None else 0
        self.visited_targets: List[int] = visited_targets if visited_targets is not None else []
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
        self.visited_targets.sort()
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


class State:
    def __init__(self, current_pos, visited_cities) -> None:
        self.Current = current_pos if current_pos is not None else 0
        self.Visited = sorted(visited_cities).copy() if visited_cities is not None else []

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return (self.Current == other.Current) and (self.Visited == other.Visited)
    def __hash__(self):
        return hash((self.Current, tuple(self.Visited)))
    def __repr__(self):
        return f"State(Current={self.Current}, Visited={self.Visited})"




def generate_power_set(n):
    original_set = list(range(0, n))
    
    power_set_size = 2 ** n
    
    power_set = []
    
    for i in range(power_set_size):
        subset = []
        
        for j in range(n):
            if i & (1 << j):
                subset.append(original_set[j])
        
        power_set.append(subset)

    return power_set


def initialize_value_and_policy(num_targets):
    stateSpace = []
    policy : dict[State] = {}
    values : dict[State] = {}
    # init_state = State(0, [])
    # policy[init_state] = 0
    visited_list = generate_power_set(num_targets)
    for target in range(num_targets):
        for visited in visited_list:     
            if target == 0 and visited == [0]:       
                pass
            # if (target in visited):
            state = State(target, visited)
            stateSpace.append(state)
            policy[state] = 0
            values[state] = 0

    return stateSpace, policy, values

def value_iteration(env: TSP, epsilon : float, gamma : float):
    statespace, policy, values = initialize_value_and_policy(num_targets)
    isTerminated = False
    while (not isTerminated):
        isTerminated = True
        VNew = {}

        for state in statespace:
            maxValue = float('-inf')
            bestAction = 0
            for action in range(env.num_targets):
                obs = env.reset(initial_state=state.Current, visited_targets=state.Visited)
                if action not in state.Visited:
                    obs_, reward, terminated, truncated, info = env.step(action)
                    visited = state.Visited
                    newstate = State(env.loc, visited)
                    if newstate.Current == 0 and newstate.Visited == [0]:
                        pass
                    action_value = reward + gamma * values[newstate]

                    if (action_value > maxValue):
                        maxValue = action_value
                        bestAction = action
            VNew[state] = maxValue
            policy[state] = bestAction

            if (np.abs(VNew[state] - values[state]) > epsilon):
                isTerminated = False
    
        values = VNew

    return statespace, values, policy


if __name__ == "__main__":
    num_targets = 6
    episodes = 1
    gamma = 0.9
    max_steps = 100

    env = TSP(num_targets)
    obs = env.reset()
    ep_rets = []

    # statespace, values, policy = value_iteration(env, 1e-6, 0.9)
    policy = {}
    Qvalue = {}


    for ep in range(episodes):
        ret = 0
        obs = env.reset(initial_state=0, visited_targets=[0])
        
        for step in range(max_steps):
            state = State(env.loc, env.visited_targets)
            action = (
                
            )  # You need to replace this with your algorithm that predicts the action.
    
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += reward

            if done:
                break

        ep_rets.append(ret)
        print(f"Episode {ep} : {ret}")

    print(np.mean(ep_rets))
