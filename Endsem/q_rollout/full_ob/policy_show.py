
from environment import PressurePlate
import gym
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import randomcolor

iteration = 10    # Total number of episodes to run
max_steps = 700         # Max steps per episode
n_actions = 5            # Number of possible actions each agent can take
a_range = 1
a_co = ((2 * a_range + 1) ** 2) * 4

filepath = "policy/4_(28, 16)_q_roll_policy.npy"

policy = np.load(filepath)

shape = policy.shape

gridsize = (shape[2],shape[1])
# Size of the grid environment (28,16) for max
n_agents = shape[0]

def policy_eval(state):
    actions = []
    for i in range(n_agents):
        x, y = state[i][0], state[i][1]
        actions.append(policy[i, x, y])
    return actions

def main():
    env = PressurePlate(gridsize[0], gridsize[1], n_agents, a_range, 'linear')

    obs_ = env.reset()
    
    for it in range(iteration):
        obs_ = env.reset()
        state = np.array([obs_[i][a_co:a_co + 2] for i in range(n_agents)], dtype=int)
       
        for step in range(max_steps):
            
            for agent in range(n_agents):
                
                temp_act = [4] * n_agents
                
                action = policy_eval(state) 
                
                temp_act[agent] = action[agent]
                
                obs, rewards, done, _ = env.step(action)
                
                state = np.array([obs[i][a_co:a_co + 2] for i in range(n_agents)], dtype=int)
            
            
                env.render()
                time.sleep(0.5)
            
            if all(done):
                break
        
main()

