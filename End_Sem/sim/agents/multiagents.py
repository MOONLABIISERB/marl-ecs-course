import math
import random
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.nn import functional as F
from torch.optim import Adam

from model.dqn import DQNUnit
from utils.config import Config

config = Config('./config')


def hard_update(target, policy):
    """
    Copy network parameters from source to target
    """
    for target_param, param in zip(target.parameters(), policy.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, policy, tau=config.learning.tau):
    for target_param, param in zip(target.parameters(), policy.parameters()):
        target_param.data.copy_(target_param.data * tau + param.data * (1. - tau))


class Agent:
    type = "prey"  # or predator
    id = 0
    # For RL
    gamma = 0.9
    EPS_START = 0.01
    lr = 0.1
    update_frequency = 0.1
    update_type = "hard"

    def __init__(self, type, agent_id, device, agent_config):
        assert type in ["prey", "predator"], "Agent type is not correct."
        self.type = type
        self.id = agent_id
        self.memory = None
        self.number_actions = 7 if config.env.world_3D else 5

        # For RL
        self.gamma = agent_config.gamma
        self.EPS_START = agent_config.EPS_START
        self.EPS_END = agent_config.EPS_END
        self.EPS_DECAY = agent_config.EPS_DECAY
        self.lr = agent_config.lr
        self.update_frequency = agent_config.update_frequency
        assert agent_config.update_type in ["hard", "soft"], "Update type is not correct."
        self.update_type = agent_config.update_type

        self.colors = {"prey": "#a1beed", "predator": "#ffd2a0"}

        self.device = device

    def draw_action(self, observation, no_exploration=False):
        raise NotImplementedError

    def update(self, *params):
        if self.update_type == "hard":
            hard_update(*params)
        elif self.update_type == "soft":
            soft_update(*params)

    def plot(self, position, agent_type, reward, radius, ax: Union[plt.Axes, Axes3D]):
        if len(position) == 2:
            x, y = position
            circle = plt.Circle((x, y), radius=radius, color=self.colors[agent_type])
            ax.add_artist(circle)
            ax.text(x - radius / 2, y, self.id)
            ax.text(x - radius / 2, y - 0.05, "Reward: {}".format(round(reward, 3)))
        else:  # 3D
            x, y, z = position
            ax.scatter(x, y, z, s=radius, c=self.colors[agent_type], marker="o")
            ax.text(x, y, z, self.id + "(r = {})".format(round(reward, 3)))
            ax.set_zlim3d(0, 1)
            ax.set_ylim3d(0, 1)
            ax.set_xlim3d(0, 1)

    def learn(self, batch, *params):
        raise NotImplementedError

    def save(self, name):
        raise NotImplementedError

    def load(self, name):
        raise NotImplementedError


class AgentDQN(Agent):
    def __init__(self, type, agent_id, device, agent_config):
        super(AgentDQN, self).__init__(type, agent_id, device, agent_config)

        self.policy_net = DQNUnit().to(self.device)
        self.target_net = DQNUnit().to(self.device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=config.agents.lr)
        self.update(self.target_net, self.policy_net)
        self.target_net.eval()

        self.n_iter = 0
        self.steps_done = 0

    def draw_action(self, state, no_exploration=False):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        with torch.no_grad():
            p = np.random.random()
            state = torch.tensor(state).to(self.device).unsqueeze(dim=0)
            if no_exploration or p > eps_threshold:
                action_probs = self.policy_net(state).detach().cpu().numpy()
                action = np.argmax(action_probs[0])
            else:
                action = random.randrange(self.number_actions)
            return action

    def load(self, name):
        params = torch.load(name)
        self.policy_net.load_state_dict(params['policy'])
        self.target_net.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])

    def save(self, name):
        save_dict = {'policy': self.policy_net.state_dict(),
                     'target_policy': self.target_net.state_dict(),
                     'policy_optimizer': self.policy_optimizer.state_dict()}
        torch.save(save_dict, name)

    def learn(self, batch):
        state_batch, next_state_batch, action_batch, reward_batch = batch
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)

        action_batch = action_batch.reshape(action_batch.size(0), 1)
        reward_batch = reward_batch.reshape(reward_batch.size(0), 1)

        policy_output = self.policy_net(state_batch)
        action_by_policy = policy_output.gather(1, action_batch)

        if config.learning.DDQN:
            actions_next = self.policy_net(next_state_batch).detach().max(1)[1].unsqueeze(1)
            Qsa_prime_targets = self.target_net(next_state_batch).gather(1, actions_next)
        else:
            Qsa_prime_targets = self.target_net(next_state_batch).detach().max(1)[0].unsqueeze(1)

        actions_by_cal = reward_batch + (self.gamma * Qsa_prime_targets)

        loss = F.mse_loss(action_by_policy, actions_by_cal)
        self.policy_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

        if not self.n_iter % self.update_frequency:
            self.update(self.target_net, self.policy_net)

        self.n_iter += 1

        return loss.detach().cpu().item()
