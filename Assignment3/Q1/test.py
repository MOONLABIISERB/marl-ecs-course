# %%
from environment import MAPFEnvironment
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from rollout import Q_learning
# Correct example setup
grid_size = (10, 10)  # 9x9 grid
obstacles = [(0, 4), (1, 4), (2, 4), (2, 5), (4, 0), (4, 1), (4, 2), (5, 2), (4, 7), (4, 8), (4, 9), (5, 7), (7, 5), (8, 5), (9, 5), (7, 4)]  # List of tuples for obstacle positions
agents = {1: (1, 1), 2: (1, 8), 3: (8, 1), 4: (8, 8)}  # Starting positions of agents
goals = {1: (5, 8), 2: (8, 4), 3: (1, 5), 4: (5, 1)}  # Goal positions of agents
env = MAPFEnvironment(grid_size, obstacles, agents, goals)
roll = [Q_learning(grid_size, env.action_space.n) for i in range(len(agents))]
Q = np.load("q_table.npy", allow_pickle=True)
count = 0
obs = env.reset()
max_steps = 100
while count < max_steps:
    actions = tuple(roll[i].best_action(Q[i], obs[i]) for i in range(len(agents)))
    obs_, reward, terminated, truncated, _ = env.step(actions)
    obs = obs_
    done = terminated
    count += 1
    env.render()
    if all(done):
        break
    plt.pause(0.7)
env.render()
print("Time taken in steps: ", count)    


