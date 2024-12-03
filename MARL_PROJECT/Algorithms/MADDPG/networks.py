import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, fc1_dims, fc2_dims, 
                    name=None, chkpt_dir=None):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        #do make sure that action is also channel to the convolution layer
        self.conv1=nn.Conv2d(8,4,3,stride=1,padding='valid')
        self.fc1 = nn.Linear(36, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, 64)
        self.q = nn.Linear(64, 1)
        self.flatten=nn.Flatten()

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        state=T.permute(state,(0,3,1,2))
        x=self.conv1(state)
        x=self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, fc2_dims=64, 
                  name='a', chkpt_dir='a'):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.conv1=nn.Conv2d(8,4,3,stride=1,padding='valid')
    
        self.fc1 = nn.Linear(36, 64)
        self.pi = nn.Linear(64, 25)
        self.flatten=nn.Flatten()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        state=T.permute(state,(0,3,1,2))
        x=self.conv1(state)
        x=self.flatten(x)
        x = F.relu(self.fc1(x))      #changed here
        pi = T.softmax(self.pi(x), dim=1)
        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

