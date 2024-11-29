import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


#P:\MARL_project\Reinforcement-Learning-for-Chain-Reaction-Game\Algorithms\PPO\
class ActorNetwork(nn.Module):
    def __init__(self,  alpha,name,
            fc1_dims=64, fc2_dims=64, chkpt_dir='checkpoints'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name,'actor_torch_ppo')
        self.conv1=nn.Conv2d(8,4,3,stride=1,padding='valid')
    
        self.fc1 = nn.Linear(36, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, 25)
        self.flatten=nn.Flatten()
        self.relu=nn.ReLU()
        self.soft=nn.Softmax(dim=-1)
        

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state=T.permute(state,(0,3,1,2))
        dist = self.conv1(state)
        dist = self.flatten(dist)
        dist = self.fc1(dist)
        dist = self.relu(dist)
        dist = self.fc2(dist)
        dist = self.relu(dist)
        dist = self.pi(dist)
        dist = self.soft(dist)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
#P:\MARL_project\Reinforcement-Learning-for-Chain-Reaction-Game\Algorithms\PPO\
class CriticNetwork(nn.Module):
    def __init__(self,  alpha, name,fc1_dims=64, fc2_dims=64,
            chkpt_dir='checkpoints'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name,'critic_torch_ppo')
        self.conv1=nn.Conv2d(8,4,3,stride=1,padding='valid')
        self.fc1 = nn.Linear(36, fc1_dims)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        self.flatten=nn.Flatten()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state=T.permute(state,(0,3,1,2))
        value = self.conv1(state)
        value = self.flatten(value)
        value = self.fc1(value)
        value = self.relu(value)
        value = self.fc2(value)
        value = self.relu(value)
        value = self.q(value)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

