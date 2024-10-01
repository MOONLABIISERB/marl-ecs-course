"""Environment for Modified Travelling Salesman Problem."""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from numpy import typing as npt
import random 
import matplotlib.pyplot as plt
from collections import Counter
# from sarsa import SARSA


class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits.

    The agent navigates a set of targets based on precomputed distances. It aims to visit
    all targets so maximize profits. The profits decay with time.
    """

    def __init__(
        self,
        num_targets: int = 10,
        max_area: int = 15,
        shuffle_time: int = 10,
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

def epsilon_greedy(Q, state, epsilon):
    state = state
    if random.random() > epsilon:
        return np.argmax(Q[state])
    else:
        return random.randint(0, 9)  # You need to replace this with your algorithm that randomly selects an action.

def path (Q, state):
    visited_states = []
    for i in range(10) :
        next_state = np.argmax(Q[state])
        
        visited_states.append(next_state)
        state = next_state
    
    visited_states = np.array(visited_states, dtype=int)
    # print(visited_states)
    
    return visited_states

def plot_graph(cumulative_rewards, episode_rewards, n_episodes):
    episodes = np.arange(1, len(cumulative_rewards)+1)  # 100 episodes
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_rewards, label='Episodic Reward', color='g', alpha=0.5)

    plt.plot(episodes, cumulative_rewards, label='Cumulative Reward', color='b')
    plt.title('Episode vs Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid()
    plt.legend()
    plt.savefig("output.jpg")
    plt.show()
    

def main() -> None:
    """Main function."""
    num_targets = 10 #num of targets
       
    env = ModTSP(num_targets)
    
    obs = env.reset() # initialize environment
    
    #initialize Q-value
    Q = np.zeros((num_targets,num_targets), dtype= float)
        
    ep_rets = [] #episodic rewards lsit
    cum = [] #cumilative rewards list
    
    n_episodes = int(4e4) #number of episodes
    
    alpha = 0.001 #laerning rate
    gamma = 0.9 #penalty
    decay = 1.01
    1 #decay function
    epsilon = 0.9 # espsilon for spsilon greedy policy
    threshold  = 2e4 #converging threshold for episodic reward value
    
    for ep in range(n_episodes):
        ret = 0 #total rewards
        obs = env.reset() #reset all initial reward,visited_states,distance etc
       
       
        state = int(obs[0][0])  # You need to extract the current state from the observation.
        epsilon = epsilon**(decay**(ep/1000)) # decaying epsilon value for more exploration in start and less in end
    
        for _ in range(100):              
            
            action = epsilon_greedy(Q,state,epsilon) # using epsilon_greedy for calculating  action
            
            obs_, reward, terminated, truncated, info = env.step(action) # returns next state and reward by taking an action
            
            next_state = int(obs_[0]) #extract location from the state
            # print(next_state)
            
            # update Q values
            Q[state][action] += alpha * (reward + gamma*(Q[next_state][action]) - Q[state][action]) # using SARSA
            
            
            
            done = terminated or truncated # if reached maximum steps
            ret += reward
            obs = obs_ # update current state as next state
            state = next_state # update current location as next location
            
            if done: # if reached max steps terminate
                break
        
                
        ep_rets.append(ret) #add total reward to episodic_reward list
        mean = np.mean(ep_rets) #mean of array of total rewards per episode
        cum.append(mean) # add mean total reward to episodic_reward list
        
        counter = Counter(ep_rets)
        # Get the most common element and its frequency
        converged_value, frequency = counter.most_common(1)[0]
        
        # Check if the value is converged or not
        if frequency > threshold:
            print(f"Converged at Episode : {ep} with reward value : {converged_value}") #if converged break out of loop
            break
        
        print(f"Episode {ep} : {ret}") #prints current episodic reward
        print(f"Episode {ep} : {mean}") #prints current mean episodic reward

    print(f"Converged at Episode : {ep} with reward value : {converged_value}")
    #  Plots episodic and cumilative rewards at each episode 
    paths = path(Q, 0)
    print(f"path travelled : {paths}")
    
    plot_graph(cum, ep_rets, n_episodes)
    

if __name__ == "__main__":
    main()
