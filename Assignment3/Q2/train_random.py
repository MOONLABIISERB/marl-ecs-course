# %%
from environment import MAPFEnvironment
import time
import numpy as np
import matplotlib.pyplot as plt
import random
print(np.random.randint(0, 5))
from rollout import Q_learning
# Correct example setup
grid_size = (10, 10)  # 9x9 grid
obstacles = [(0, 4), (1, 4), (2, 4), (2, 5), (4, 0), (4, 1), (4, 2), (5, 2), (4, 7), (4, 8), (4, 9), (5, 7), (7, 5), (8, 5), (9, 5), (7, 4)]  # List of tuples for obstacle positions
goals = {1: (5, 8), 2: (8, 4), 3: (1, 5), 4: (5, 1)}  # Goal positions of agents
# %%
#Initialize the environment
# def generate_random_position(grid_size, obstacles):
#     while True:
#         x = random.randint(0, grid_size[0] - 1)
#         y = random.randint(0, grid_size[1] - 1)
#         if (x, y) not in obstacles:
#             return (x, y)

# agents = {i: generate_random_position(grid_size, obstacles) for i in range(1, 5)}

def generate_random_position(grid_size, obstacles):
    while True:
        x = random.randint(0, grid_size[0] - 1)
        y = random.randint(0, grid_size[1] - 1)
        if (x, y) not in obstacles:
            return (x, y)


# Generate positions for 4 agents, ensuring no overlap
agents = {}
used_positions = set()

for i in range(1, 5):
    # Keep generating positions until we get a unique one
    while True:
        position = generate_random_position(grid_size, obstacles)
        if position not in used_positions:
            used_positions.add(position)
            agents[i] = position
            break


env = MAPFEnvironment(grid_size, obstacles, agents, goals)
roll = [Q_learning(grid_size, env.action_space.n) for i in range(len(agents))]
obs = env.reset()
print(env.action_space.n)
env.render()
# %%
Q = [roll[i].create_q() for i in range(len(agents))]
# %%
n_iterations = 4000
n_steps = 300
min_epsilon = 0.2
epsilon = 1.0
decay_rate = 0.999
alpha = 0.05
gamma = 0.95
for i in range(n_iterations):
    agents = {}
    used_positions = set()

    for t in range(1, 5):
        # Keep generating positions until we get a unique one
        while True:
            position = generate_random_position(grid_size, obstacles)
            if position not in used_positions:
                used_positions.add(position)
                agents[t] = position
                break
    env = MAPFEnvironment(grid_size, obstacles, agents, goals)
    obs = env.reset()
    count = 0
    for j in range(n_steps):
        if i < 500:
            epsilon = 0.9
        if i < 1000 and i >= 500:
            epsilon = 0.8
        elif i>=1000 and i<1500:  
            epsilon = 0.7
        elif i>=1500 and i<2000:
            epsilon = 0.6      
        else:    
            epsilon = max(min_epsilon, epsilon*decay_rate)
        # if j == 0:
        #     actions = tuple(random.randint(0, env.action_space.n-1) for i in range(len(agents)))
        # else:    
        actions = tuple(roll[i].epsilon_greedy(obs[i], epsilon, Q[i]) for i in range(len(agents)))                    
        obs_, reward, terminated, truncated, _ = env.step(actions)
        done = terminated
        count += 1
        for k in range(len(agents)):
            Q[k][(obs[k], actions[k])] += alpha * (reward[k] + gamma * roll[k].get_max(Q[k], obs_[k]) - Q[k][(obs[k], actions[k])])
        if all(done):            
            break
        obs = obs_
        #env.render()        
        #plt.pause(0.7)   
    print(f"Episode {i} : {count}  epsilon: {epsilon}")   
env.render()     
np.save("q_table_random.npy", Q)        



# Perform a step
# actions = (2, 2, 1, 1)  # Example actions
# obs_, reward, terminated, truncated, _ = env.step(actions)
# env.render()

# %%
