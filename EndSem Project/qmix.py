import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from environment import MultiCarRacing
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import json

# Setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiencing replay memory tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ).to(device)
    
    def forward(self, x):
        return self.network(x)

class MixingNetwork(nn.Module):
    def __init__(self, n_agents, hidden_dim):
        super(MixingNetwork, self).__init__()
        self.mixing_network = nn.Sequential(
            nn.Linear(n_agents * 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
    
    def forward(self, agent_q_values):
        return self.mixing_network(agent_q_values.view(-1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class QMIX:
    def __init__(self, n_agents, state_dim, action_dim, writer, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99, gamma=0.95, batch_size=64, target_update_frequency=10):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.writer = writer
        
        # Creating networks and replay buffers for each agent
        self.q_networks = {}
        self.target_networks = {}
        self.mixing_network = MixingNetwork(n_agents, hidden_dim=128)
        self.optimizers = {}
        self.replay_buffers = {}
        
        for agent_id in range(n_agents):
            self.q_networks[agent_id] = QNetwork(state_dim, action_dim)
            self.target_networks[agent_id] = QNetwork(state_dim, action_dim)
            self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())
            self.optimizers[agent_id] = optim.Adam(self.q_networks[agent_id].parameters())
            self.replay_buffers[agent_id] = ReplayBuffer(capacity=10000)
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.update_counter = 0
        self.training_step = 0
    
    def select_action(self, states):
        actions = {}
        for agent_id in range(self.n_agents):
            if random.random() < self.epsilon:
                actions[agent_id] = random.randint(0, self.action_dim - 1)
            else:
                state = torch.FloatTensor(states[agent_id]).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = self.q_networks[agent_id](state)
                    actions[agent_id] = q_values.argmax().item()
        return actions
    
    def update(self):
        if any(len(self.replay_buffers[agent_id]) < self.batch_size for agent_id in range(self.n_agents)):
            return
        
        # Sampling a batch of experiences
        batch = [self.replay_buffers[agent_id].sample(self.batch_size) for agent_id in range(self.n_agents)]
        
        states = [torch.stack([batch[agent_id][0] for agent_id in range(self.n_agents)], dim=1).to(device)]
        actions = [torch.stack([batch[agent_id][1] for agent_id in range(self.n_agents)], dim=1).to(device)]
        rewards = [torch.stack([batch[agent_id][2] for agent_id in range(self.n_agents)], dim=1).to(device)]
        next_states = [torch.stack([batch[agent_id][3] for agent_id in range(self.n_agents)], dim=1).to(device)]
        dones = [torch.stack([batch[agent_id][4] for agent_id in range(self.n_agents)], dim=1).to(device)]
        
        # Computing current Q values for all agents
        q_values = [self.q_networks[agent_id](states[agent_id]) for agent_id in range(self.n_agents)]
        q_values = torch.stack(q_values, dim=1)
        
        # Computing next Q values for all the agents
        next_q_values = [self.target_networks[agent_id](next_states[agent_id]) for agent_id in range(self.n_agents)]
        next_q_values = torch.stack(next_q_values, dim=1)
        
        # Computing joint Q-value using the mixing network
        q_values_joint = self.mixing_network(q_values)
        next_q_values_joint = self.mixing_network(next_q_values)
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_joint
        
        # Computing the loss and optimizing
        loss = nn.MSELoss()(q_values_joint, target_q_values.detach())
        
        for agent_id in range(self.n_agents):
            self.optimizers[agent_id].zero_grad()
            loss.backward()
            self.optimizers[agent_id].step()
        
        # Updating target networks periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            for agent_id in range(self.n_agents):
                self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_step += 1

    def save_model(self, save_dir, episode):
        """Save model checkpoints and training state"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        checkpoint = {
            'episode': episode,
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'models': {},
            'optimizers': {}
        }
        
        for agent_id in range(self.n_agents):
            checkpoint['models'][f'agent_{agent_id}'] = self.q_networks[agent_id].state_dict()
            checkpoint['optimizers'][f'agent_{agent_id}'] = self.optimizers[agent_id].state_dict()
        
        checkpoint['mixing_network'] = self.mixing_network.state_dict()
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint_episode_{episode}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        config = {
            'n_agents': self.n_agents,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update_frequency': self.target_update_frequency
        }
        
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def load_model(self, checkpoint_path):
        """Load model checkpoints and training state"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        for agent_id in range(self.n_agents):
            self.q_networks[agent_id].load_state_dict(checkpoint['models'][f'agent_{agent_id}'])
            self.optimizers[agent_id].load_state_dict(checkpoint['optimizers'][f'agent_{agent_id}'])
        
        self.mixing_network.load_state_dict(checkpoint['mixing_network'])
        
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        print(f"Loaded checkpoint from {checkpoint_path}")
