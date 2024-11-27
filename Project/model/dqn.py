import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import Config

config = Config('./config')

# DQN Q-Network
class DQNUnit(nn.Module):
    def __init__(self):
        super(DQNUnit, self).__init__()

        n_actions = 7 if config.env.world_3D else 5
        self.n_agents = config.agents.number_preys + config.agents.number_predators
        n_obstacles = 2 * len(config.env.obstacles)
        n_magic_switch = int(config.env.magic_switch) * (2 + self.n_agents)

        self.fc = nn.Sequential(
            nn.Linear(self.n_agents * 3 + n_obstacles + n_magic_switch, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),  # n_actions depends on the environment
        )

    def forward(self, x):
        return self.fc(x)


# DQN Critic Network
class DQNCritic(nn.Module):
    def __init__(self):
        super(DQNCritic, self).__init__()

        action_dim = 7 if config.env.world_3D else 5
        n_agents = config.agents.number_preys + config.agents.number_predators
        n_obstacles = 2 * len(config.env.obstacles)
        state_dim = n_agents * 3 + n_obstacles + int(config.env.magic_switch) * (2 + n_agents)

        self.fc = nn.Sequential(
            nn.Linear(state_dim + n_agents * action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Q-value prediction
        )

    def forward(self, x, actions):
        """
        Args:
            x: (batch_size, state_size)
            actions: [(batch_size, action_size)] list size n_agents
        Returns:
            Q-value prediction
        """
        x = torch.cat([x, *actions], dim=1)  # Concatenate state and actions
        return self.fc(x)


# DQN Actor Network (for Gumbel-Softmax sampling)
class DQNActor(nn.Module):
    def __init__(self):
        super(DQNActor, self).__init__()

        action_dim = 7 if config.env.world_3D else 5
        n_agents = config.agents.number_preys + config.agents.number_predators
        n_obstacles = 2 * len(config.env.obstacles)
        state_dim = n_agents * 3 + n_obstacles + int(config.env.magic_switch) * (2 + n_agents)

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),  # Action probabilities
        )

    def forward(self, x):
        # Use Gumbel-Softmax for differentiable action sampling
        return F.gumbel_softmax(self.fc(x), tau=config.learning.gumbel_softmax_tau)
