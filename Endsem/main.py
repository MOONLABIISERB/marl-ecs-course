from environment import MultiCarRacing
import numpy as np


env = MultiCarRacing(n_cars=4, grid_size=30, track_width=5, render_mode='human')
env.reset()
print(env.agents[0].observation)
while True:
    actions = {agent_id: np.random.randint(0,5) for agent_id in range(4)}  # random
    obs, rewards, dones, info = env.step(actions)
    env.render()
    
    if any(dones.values()):
        break