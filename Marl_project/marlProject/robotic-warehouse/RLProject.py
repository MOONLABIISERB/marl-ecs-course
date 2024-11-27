#%%
import gymnasium as gym
import rware
import time
from rware import RewardType
from rware import ObservationType
# %%
layout = """
......
..xx..
..xx..
..xx..
......
..gg..
"""
env = gym.make("rware-tiny-2ag-v2", layout=layout, reward_type=RewardType.TWO_STAGE)
# print(env.fast_obs)
# print(env.n_agents)
# print()
# print(env.observation_space[0].shape[0])
# print("Observation space", env.observation_space)
# print("Action Space", env.action_space)
# print("n actions", env.action_space[0].n)
observation, info = env.reset()

env.render()
# %%

# %%
env.reset()
time.sleep(3)
print("Observation Space:", env.observation_space)
while True:
    actions = env.action_space.sample()  # the action space can be sampled
    n_obs, reward, done, truncated, info = env.step(actions)
    #print("done: ", done)
    if reward[0]!=-0.4 or reward[1]!=-0.4:
        print(reward)
    #reward1 = int(reward[0])
    # print(f"{actions}\n")
    # for agent_id in range(env.n_agents):
    #     print(f"Agent {agent_id} observation:", n_obs[agent_id]) 
        #print(reward1 + 1)      
    env.render()
    time.sleep(0.2)
env.close()
# %%
env.close()

# %%
