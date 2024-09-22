"""Environment for Travelling Salesman Problem."""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from collections import defaultdict

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
        self.clocks: np.ndarray = np.zeros(self.num_targets)
        self.dist: List = self.distances[self.loc]

        state = np.concatenate(
            (
                np.array([self.loc]),
                np.array(self.clocks),
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


        self.distances[self.loc, next_loc]

        reward = self._get_rewards(past_loc, next_loc)

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
        self.visited_targets.append(next_loc)# change made to order of reward assignment

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

                                                    ########################## New Code Here #################################
if __name__ == "__main__":
    num_targets = 50 # set number of states

    env = TSP(num_targets)
    obs = env.reset()
    ep_rets = []

    first_visit=False #set to True if first_visit MC strategy has to be executed,set to False for every_visit MC strategy 
    q = defaultdict() # dict for averaging returns
    Returns = defaultdict(list) #returns dict
    policy = np.array(np.arange(num_targets)) # random policy

    for ep in range(2000):
        obs = env.reset()
        episode = []
        ret = 0

        env.loc = env.action_space.sample()
        for k in range(int(num_targets)):          
                                                    # generating episodes
            if k==0:                                # initial state action pair randomly generated                                                                                 
                action = (
                    env.action_space.sample()
                )

                episode.append([int(env.loc),int(action)])
                obs_, reward, terminated, truncated, info = env.step(action)
                episode[k].append(reward)
                
            else:
                action = (                      # following policy after first random action
                    policy[int(env.loc)]
                )
                episode.append([int(env.loc),int(action)])
                obs_, reward, terminated, truncated, info = env.step(action)
                episode[k].append(reward)

        g = 0
        gamma = 1
        for t in range(len(episode)-1,-1,-1):

            g = gamma*g + episode[t][2]

            if first_visit:                                   # tracking previous states for first visit strategy
                prev_states = []
                for m in range(t-1):
                    prev_states.append(episode[m][0])

                if episode[t][0] in prev_states:
                    continue
            if len(Returns[(episode[t][0],episode[t][1])])==0 :
                Returns[(episode[t][0],episode[t][1])] = [0,0]                    

            Returns[(episode[t][0],episode[t][1])] = [sum(x) for x in zip([g,1], Returns[(episode[t][0],episode[t][1])])] # saving values to returns
            ret += g
        ep_rets.append(ret)
        print(f"Episode {ep} : {ret}")

        for key,value in Returns.items():
            q[key] = value[0]/value[1] 
                                                        # averaging returns
            dictkeys = list(q.keys())
            dictvalues = list(q.values())
            sorted_value_index = np.argsort(dictvalues)[::-1]
            sorted_action_value = {dictkeys[i]: dictvalues[i] for i in sorted_value_index}
            
        best_actions = defaultdict()                    
        for k,val in sorted_action_value.items():
            previous_policy = policy
            if k[0] in best_actions.keys():             # picking best action for state from the averged returns 
                continue
            best_actions[k[0]] = k[1]
            policy[k[0]] = k[1]

        
    print(len(q),"combinations of state-action pairs explored out of:",num_targets**2)
    print(policy)
