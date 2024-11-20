# %%
from environment import MAPFEnvironment
import time
import numpy as np
import matplotlib.pyplot as plt
import random
print(np.random.randint(0, 5))
from rollout import Q_learning

def plot_learning_curve(x, scores, filename):
    plt.figure()
    plt.plot(x, scores, label='Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(filename)
    plt.show()

# Correct example setup
grid_size = (10, 10)  # 9x9 grid
obstacles = [(0, 4), (1, 4), (2, 4), (2, 5), (4, 0), (4, 1), (4, 2), (5, 2), (4, 7), (4, 8), (4, 9), (5, 7), (7, 5), (8, 5), (9, 5), (7, 4)]  # List of tuples for obstacle positions
agents = {1: (1, 1), 2: (1, 8), 3: (8, 1), 4: (8, 8)}  # Starting positions of agents
goals = {1: (5, 8), 2: (8, 4), 3: (1, 5), 4: (5, 1)}  # Goal positions of agents
# %%
# Initialize the environment
env = MAPFEnvironment(grid_size, obstacles, agents, goals)
roll = [Q_learning(grid_size, env.action_space.n) for i in range(len(agents))]
obs = env.reset()
print(env.action_space.n)
env.render()
# %%
Q = [roll[i].create_q() for i in range(len(agents))]
# %%
n_iterations = 1000
n_steps = 100
min_epsilon = 0.001
epsilon = 1.0
decay_rate = 0.9995
alpha = 0.15
gamma = 0.95
scores = []
for i in range(n_iterations):
    obs = env.reset()
    count = 0
    episode_scores = 0
    for j in range(n_steps):
        
        epsilon = max(min_epsilon, epsilon*decay_rate)
        # if j == 0:
        #     actions = tuple(random.randint(0, env.action_space.n-1) for i in range(len(agents)))
        # else:    
        actions = tuple(roll[i].epsilon_greedy(obs[i], epsilon, Q[i]) for i in range(len(agents)))                    
        obs_, reward, terminated, truncated, _ = env.step(actions)
        episode_scores += 1
        done = terminated
        count += 1
        for k in range(len(agents)):
            Q[k][(obs[k], actions[k])] += alpha * (reward[k] + gamma * roll[k].get_max(Q[k], obs_[k]) - Q[k][(obs[k], actions[k])])
        if all(done):            
            break
        obs = obs_
        #env.render()        
        #plt.pause(0.7)
    scores.append(episode_scores)    
    #env.render()    
    print(f"Episode {i} : {count}  epsilon: {epsilon}")   
#np.save("q_table.npy", Q)  
filename = f'steps_.png'
x = [i+1 for i in range(n_iterations)]
plot_learning_curve(x, scores, filename)





# Perform a step
# actions = (2, 2, 1, 1)  # Example actions
# obs_, reward, terminated, truncated, _ = env.step(actions)
# env.render()

# %%
