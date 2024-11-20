"""Environment for MAPF."""
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from visualizer import gif_render
class MAPF(gym.Env):

    def __init__(self, seed: int = None, render = False) -> None:
        """Initialize the environment.
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed=seed)
        self.rendering = render
        self.steps: int = 0
        self.max_steps: int = None
        self.rewards = None
        self.walls = [(4,0),(4,1),(4,2),(5,2),(0,4),(1,4),(2,4),(2,5),(7,5),(7,4),(8,4),(9,4),(4,7),(5,7),(5,8),(5,9)]
        self.agents = ["green","purple","blue","yellow"]
        self.agent_id = {'green':0,'purple':1,'blue':2,'yellow':3}
        self.goals = {'green':(1,1),'purple':(8,1),'blue':(1,8),'yellow':(8,8)}
        self.agent_loc = {'green':(8,5),'purple':(4,8),'blue':(5,1),'yellow':(1,5)}
        self.infos = {name: {} for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.state = None
    def observation_space(self, agent):

        self.observation_spaces = {
            name: spaces.Dict({"observation": spaces.Box(low=0, high=1, shape=(10, 10), dtype=bool)}
                              ) for name in self.agents}
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        self.action_spaces = {name: spaces.Discrete(5) for name in self.agents}
        return self.action_spaces[agent]

    def reset(self, seed: Optional[int] = None): 
        self.steps: int = 0
        self.max_steps: int = 10000
        self.rewards = {name: 0 for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.goals = {'green':(1,1),'purple':(8,1),'blue':(1,8),'yellow':(8,8)}
        self.agent_startloc = {'green':(8,5),'purple':(4,8),'blue':(5,1),'yellow':(1,5)}      
        self.agent_currentloc = self.agent_startloc
        goalmap = np.zeros(shape=(10,10,4))
        for goal in self.goals.keys():
            goalmap[self.goals[goal][0],self.goals[goal][1]] = 1
        self.state = self.agent_currentloc
        return self.state

    def step(self, actions):
        self.steps += 1
        validity = self.update_positions(actions)
        reward = self._get_rewards(validity)
        terminated = bool(self.steps == self.max_steps)
        self.state = self.agent_currentloc
        for agent in self.agents:
            if self.agent_currentloc[agent] == self.goals[agent]:
                self.terminations[agent] = True

        return (self.state, reward, terminated, {})

    def _get_rewards(self, validity):
        for agent in self.agents:
            if validity[agent] == 0:
                self.rewards[agent] = -1
            elif self.agent_currentloc[agent] != self.goals[agent]:
                self.rewards[agent] = -1
            if self.agent_currentloc[agent] == self.goals[agent]:
                self.rewards[agent] = 0
                print(f'agent {agent} reached it\'s goal!->{self.agent_currentloc[agent]}')
        return self.rewards
    
    def update_positions(self, actions):
        #actions = 4 len list
        #green = actions[0]
        #purple = actions[1]
        #blue = actions[2]
        #yellow = actions[3]
        #[up,right,down,left,stay]=[0,1,2,3,4]
        validity = {'green':1,'purple':1,'blue':1,'yellow':1}
        for agent in self.agents:
            update = 0
            occupied = self.agent_currentloc.values()
            action = actions[self.agent_id[agent]]
            if (action == 0) & (self.agent_currentloc[agent][1]>0):
                update = (self.agent_currentloc[agent][0],self.agent_currentloc[agent][1]-1)
            else:
                validity[agent] = 0
            if (action == 1) & (self.agent_currentloc[agent][0]<9):
                update = (self.agent_currentloc[agent][0]+1,self.agent_currentloc[agent][1])
            else:
                validity[agent] = 0
            if (action == 2) & (self.agent_currentloc[agent][1]<9):
                update = (self.agent_currentloc[agent][0],self.agent_currentloc[agent][1]+1)
            else:
                validity[agent] = 0
            if (action == 3) & (self.agent_currentloc[agent][0]>0):
                update = (self.agent_currentloc[agent][0]-1,self.agent_currentloc[agent][1])
            else:
                validity[agent] = 0
            if (update not in self.walls) & (update !=0): # & (update not in occupied)
                self.agent_currentloc[agent] = update
            else:
                validity[agent] = 0
        return validity
        
if __name__ == "__main__":

    lr = 0.5
    gamma = 0.90
    eps_threshold = 1
    def agent_action(env):
        sample = random.random()
        global eps_threshold
        if sample > eps_threshold:
            actions=[policy[agent][env.agent_currentloc[agent][0],env.agent_currentloc[agent][1]] for agent in env.agents]
            return actions
        else:
            return list(np.random.randint(low=0,high=5,size=(4)))
        
    def Q_update(env, state, next_state, Q, actions):
        for agent in env.agents:
            q_now = Q[agent][state[agent][0],state[agent][1],actions[env.agent_id[agent]]]
            q_next = np.max(Q[agent][next_state[agent][0],next_state[agent][1],:])
            q = q_now + lr * (env.rewards[agent] + (gamma*q_next) - q_now)
            Q[agent][state[agent][0],state[agent][1],actions[env.agent_id[agent]]] = q
            policy[agent][state[agent][0],state[agent][1]] = np.argmax(Q[agent][next_state[agent][0],next_state[agent][1],:]).item()


    agents = ["green","purple","blue","yellow"]
    Train = False 
    if (not Train):
        env = MAPF(render=True) # set render to false to stop generating gifs and save time
        env.reset()
        ep_rets = []
        num_eps = 1
        total_steps = 1000
        if env.rendering:
            for ep in range(num_eps):
                gif_render(env=env, ep=ep, total_steps=total_steps,) # from visualizer.py
        else:
            for ep in range(num_eps):
                ret = 0
                obs = env.reset()
                for steps in range(total_steps):
                    actions = agent_action(env)
                    state, reward, terminated, info = env.step(actions)
                    if np.all([env.terminations[agent]==True for agent in env.agents]):
                        done = True
                    else:
                        done = False 
                    for agent in env.agents:
                        ret += reward[agent]
                    if done:
                        break
                ep_rets.append(ret)
                print(f"Episode Return: {ret}")
            print(f"Average Return: {np.mean(ep_rets)}")

    if Train:
        Q = {agent:np.zeros(shape=(10,10,5)) for agent in agents}
        policy = {agent:np.random.randint(0, high=5, size=(10,10), dtype=int) for agent in agents}
        num_eps = 10000
        env = MAPF(render=False)
        total_steps = 100
        ep_rets = []
        for ep in range(num_eps):
            ret = 0
            state = env.reset()
            for steps in range(total_steps):
                actions = agent_action(env)
                next_state, reward, terminated, info = env.step(actions)
                Q_update(env, state, next_state, Q, actions)
                if (np.all([env.terminations[agent]==True for agent in env.agents])) or terminated:
                    done = True
                else:
                    done = False 
                for agent in env.agents:
                    ret += reward[agent]
                if done:
                    break
                state = next_state
            ep_rets.append(ret)
            if (ep%10000 ==0) & (eps_threshold>0.2) :
                eps_threshold-=0.1
            print(f"Episode Return at ep {ep}: {ret}")
            np.save('Q_vals.npy', Q)
        print(f"Average Return: {np.mean(ep_rets)}")