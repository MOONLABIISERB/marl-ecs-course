"""Environment for Grid World Problem."""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from collections import defaultdict
import random
class grid(gym.Env):
    """Grid World setup.

    """

    def __init__(self, start_loc: tuple, end_loc: tuple, box_start_loc: tuple, world_size: tuple = (10,10), max_steps:int = 10000, seed: int = None) -> None:
        """Initialize the grid environment.

        Args:
            start_loc (tuple): start.
            end_loc (tuple): end
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed=seed)
   
        self.steps: int = 0
        self.start_loc: tuple = start_loc
        self.end_loc: tuple = end_loc
        self.max_steps: int = max_steps
        self.world_size: tuple = world_size
        self.box_start_loc: tuple = box_start_loc

        self.grid = np.zeros(self.world_size)
        
        # Action Space
        self.action_space = gym.spaces.Discrete(4)

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
        self.loc: tuple = self.start_loc
        self.box_loc: tuple = self.box_start_loc
        self.obstacles: list = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,3),(2,0),(2,3),(2,4),(2,5),(3,0),(3,5),(4,0),(4,5),(5,0),(5,3),(5,4),(5,5),
                                (6,0),(6,1),(6,2),(6,3)]
        self.corners: list = [(1,1),(1,2),(3,4),(4,4),(5,1),(5,2)]
        self.valid_starts: list = [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2),(3,3),(3,4),(4,1),(4,2),(4,4),(5,1),(5,2)]
        return {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        """Take an action (move to the next target).

        Args:
            action (int): The index of the next target to move to.

        Returns:
            Tuple [float, bool, Dict[str, None]]:
                - The reward for the action.
                - A boolean indicating whether the episode has terminated.
                - An empty info dictionary.
        """

        self.steps += 1
        next_loc, current_box_location = self.next_loc_fn(action)
        reward = self._get_rewards(next_loc, current_box_location)

        terminated = bool(self.steps == self.max_steps or current_box_location == self.end_loc or (current_box_location in self.corners))
        self.loc = next_loc
        self.box_loc = current_box_location

        return (reward, terminated, {})

    def _get_rewards(self, next_loc: tuple, box_loc: tuple,) -> float:
        """Calculate the reward based on the evnironment.

        Args:
            
            next_loc (tuple): Next location of the agent.
            box_loc (tuple): location of the box.

        Returns:
            float: Reward based on the environment constraints.
        """
        if box_loc == self.end_loc:
            reward = 0
        elif box_loc in self.corners:                       #define rewards
            reward = -1
        elif next_loc in self.obstacles:
            reward = -1
        else:
            reward = -1
        return reward
    

    def next_loc_fn(self,action):
        """Calculate the next location based on the evnironment.

        Args:
            action (int): Action of agent

        Returns:
             - next_loc,current_box_loc (int): next location of the agent, current box location.
              
        """
        #(0,1,2,3)==(up,right,down,left)
        # conditions governing the agent behavior

        past_loc = self.loc
        current_box_loc = self.box_loc
        if action == 0:
            action = (-1,0)
            next_loc = tuple(map(sum,(zip(past_loc,action))))
            box_shift = tuple(map(sum,(zip(current_box_loc,action))))
            if  current_box_loc == next_loc:
                if box_shift in self.obstacles:  
                    next_loc = past_loc
                else:
                    current_box_loc = box_shift
            elif next_loc in self.obstacles:  
                next_loc = past_loc
                
            
        elif action == 1:
            action = (0,1)
            next_loc = tuple(map(sum,(zip(past_loc,action))))
            box_shift = tuple(map(sum,(zip(current_box_loc,action))))
            if  current_box_loc == next_loc:
                if box_shift in self.obstacles:  
                    next_loc = past_loc
                else:
                    current_box_loc = box_shift
            elif next_loc in self.obstacles:  
                next_loc = past_loc

        elif action == 2:
            action = (1,0)
            next_loc = tuple(map(sum,(zip(past_loc,action))))
            box_shift = tuple(map(sum,(zip(current_box_loc,action))))
            if  current_box_loc == next_loc:
                if box_shift in self.obstacles:  
                    next_loc = past_loc
                else:
                    current_box_loc = box_shift
            elif next_loc in self.obstacles:  
                next_loc = past_loc
        else:
            action = (0,-1)
            next_loc = tuple(map(sum,(zip(past_loc,action))))
            box_shift = tuple(map(sum,(zip(current_box_loc,action))))
            if  current_box_loc == next_loc:
                if box_shift in self.obstacles:  
                    next_loc = past_loc
                else:
                    current_box_loc = box_shift
            elif next_loc in self.obstacles:  
                next_loc = past_loc
        next_loc, current_box_loc = next_loc, current_box_loc
        return next_loc, current_box_loc


if __name__ == "__main__":

    env = grid(start_loc=(1,2), end_loc=(3,1), box_start_loc=(4,3),world_size=(7,6))
    obs = env.reset()
    ep_rets = []
    first_visit=False #set to True if first_visit MC strategy has to be executed,set to False for every_visit MC strategy 
    episode_length = 100
    
    q = defaultdict()# dict for averaging returns
    Returns = defaultdict(list)#returns dict
    policy = np.full(env.world_size,fill_value=2) # random policy
    


    for ep in range(10000):
        ret = 0
        obs = env.reset()
        episode = []
        env.loc = tuple(random.sample(env.valid_starts,1))[0]

        for k in range(int(episode_length)):   # generating episodes
            if k==0:                           # initial state action pair randomly generated
                action = (
                    env.action_space.sample()
                )

                episode.append([env.loc,action])
                reward, terminated, info = env.step(action)
                episode[k].append(reward)
                
            else:
                action = (
                    policy[env.loc[0]][env.loc[1]]
                )           
                episode.append([env.loc,action])            # following policy after first random action
                reward, terminated, info = env.step(action)
                episode[k].append(reward)


        g = 0
        gamma = 1
        for t in range(len(episode)-1,-1,-1):
            g = gamma*g + episode[t][2]

            if first_visit:
                prev_states = []                         # tracking previous states for first visit strategy
                for m in range(t-1):
                    prev_states.append(episode[m][0])

                if episode[t][0] in prev_states:
                    continue
            if len(Returns[(episode[t][0],episode[t][1])])==0 :
                Returns[(episode[t][0],episode[t][1])] = [0,0]

            Returns[(episode[t][0],episode[t][1])] = [sum(x) for x in zip([g,1], Returns[(episode[t][0],episode[t][1])])]  # saving values to returns

            ret += g
        ep_rets.append(ret)
        print(f"Episode {ep} : {ret}")

        for key,value in Returns.items():
            q[key] = value[0]/value[1]                       # averaging returns

            dictkeys = list(q.keys())
            dictvalues = list(q.values())
            sorted_value_index = np.argsort(dictvalues)[::-1]
            sorted_action_value = {dictkeys[i]: dictvalues[i] for i in sorted_value_index}
        best_actions = defaultdict()                    
        for k,val in sorted_action_value.items():
            if k[0] in best_actions.keys():
                continue
            best_actions[k[0]] = k[1]
            policy[k[0][0]][k[0][1]] = k[1]             # picking best action for state from the averged returns 

    print(policy)