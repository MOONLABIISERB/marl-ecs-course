import pickle
import pygame
from environment import MultiAgentEnv
from agent import QLearningAgent
import numpy as np

np.random.seed = 100

def load_q_table(filename):
    with open(filename, 'rb') as f:
        q_table = pickle.load(f)
        return q_table

def test(env, agents, q_table):
    obs = env.reset()
    done = False

    while not done:
        actions = {}
        for agent_id, agent in agents.items():
            actions[agent_id], q_table = agent.choose_action(obs, q_table)
        
        next_obs, reward, done, _ = env.step(actions)
        
        obs = next_obs
        env.render()
        pygame.time.delay(500)  # Delay in milliseconds

agents = {
        0: (1,1),
        1: (8,1),
        2: (1,8),
        3: (8,8)
    }


goals = {
    0: (1,1),
    1: (8,1),
    2: (1,8),
    3: (8,8)
}

obstacles=[
    (0, 4),
    (1, 4),
    (2, 4),
    (2, 5),
    (4, 0),
    (4, 1),
    (4, 2),
    (5, 2),
    (4, 9),
    (4, 8),
    (4, 7),
    (5, 7),
    (9, 5),
    (8, 5),
    (7, 5),
    (7, 4)
]

num_agents = 4

epsilon = 0.1
alpha = 0.2
gamma = 0.9

q_table = load_q_table('Assignment_3/q_table_agent.pkl')

env = MultiAgentEnv(grid_size=(10, 10), goals=goals, obstacles=obstacles, num_agents=num_agents)

agents = {i: QLearningAgent(agent_id=i, action_space=env.action_spaces[i], epsilon=epsilon, alpha=alpha, gamma=gamma) for i in range(num_agents)}

test(env, agents, q_table)



        