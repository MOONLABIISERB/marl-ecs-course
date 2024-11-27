import gym
import numpy as np
from gym import spaces
from env2 import (
    DenseTetheredBoatsEnv,
)  # Assuming your environment is saved as DenseTetheredBoatsEnv.py


class SequentialActionEnv(gym.Env):
    def __init__(self, env):
        super(SequentialActionEnv, self).__init__()
        self.env = env
        self.n_agents = env.n_boats
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.current_agent = 0
        self.actions = [None] * self.n_agents

    def reset(self):
        obs = self.env.reset()
        self.current_agent = 0
        self.actions = [None] * self.n_agents
        return obs

    def step(self, action):
        self.actions[self.current_agent] = action
        if self.current_agent == self.n_agents - 1:
            # All agents have acted, perform environment step
            actions = self.actions
            obs, reward, done, info = self.env.step(actions)
            self.current_agent = 0
            self.actions = [None] * self.n_agents
            return obs, reward, done, info
        else:
            self.current_agent += 1
            # Return the current state without stepping the environment
            return self.env.grid.copy(), 0.0, False, {}

    def render(self, mode="human"):
        self.env.render(mode)

    def get_valid_actions(self):
        return self.env._get_valid_actions()
