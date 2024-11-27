import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Calculate flattened size
        self.flatten_size = 32 * input_dim * input_dim

        self.fc1 = nn.Linear(self.flatten_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        # Input shape: (batch_size, 1, grid_size, grid_size)
        x = F.relu(self.conv1(x))  # Shape: (batch_size, 16, grid_size, grid_size)
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 32, grid_size, grid_size)
        x = F.relu(self.conv3(x))  # Shape: (batch_size, 32, grid_size, grid_size)
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 32 * grid_size * grid_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, min(len(self.memory), self.batch_size))

        states = np.stack([e[0] for e in experiences])
        actions = np.vstack([e[1] for e in experiences])
        rewards = np.vstack([e[2] for e in experiences])
        next_states = np.stack([e[3] for e in experiences])
        dones = np.vstack([e[4] for e in experiences]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class CentralizedAgent:
    def __init__(
        self,
        env,
        hidden_dim=256,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update=10,
    ):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "mps"

        # Total number of actions (9 actions per boat, combined action space)
        self.n_actions = 81  # 9 * 9 for two boats

        # Q-Networks
        self.qnetwork_local = QNetwork(env.grid_size, hidden_dim, self.n_actions).to(
            self.device
        )
        self.qnetwork_target = QNetwork(env.grid_size, hidden_dim, self.n_actions).to(
            self.device
        )
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.t_step = 0

    def get_valid_actions_mask(self, state, boat_positions):
        """Generate mask for valid actions"""
        valid_actions = np.zeros(self.n_actions, dtype=np.bool_)

        # For each possible action of boat 1
        for action1 in range(9):
            new_pos1 = self.env._get_new_position(boat_positions[0], action1)

            # For each possible action of boat 2
            for action2 in range(9):
                new_pos2 = self.env._get_new_position(boat_positions[1], action2)

                # Check if both moves are valid
                if self.env._is_valid_move(
                    boat_positions[0], new_pos1, boat_positions[1]
                ) and self.env._is_valid_move(
                    boat_positions[1], new_pos2, boat_positions[0]
                ):
                    action_idx = action1 * 9 + action2
                    valid_actions[action_idx] = True

        return valid_actions

    def decode_action(self, combined_action):
        """Convert combined action index to individual boat actions"""
        action1 = combined_action // 9
        action2 = combined_action % 9
        return [action1, action2]

    def select_action(self, state, boat_positions, epsilon=None):
        """Select action using epsilon-greedy policy with action masking"""
        if epsilon is None:
            epsilon = self.epsilon

        valid_actions = self.get_valid_actions_mask(state, boat_positions)

        if random.random() < epsilon:
            # Random valid action
            valid_indices = np.where(valid_actions)[0]
            return random.choice(valid_indices)

        # Convert state to tensor (batch_size, channels, height, width)
        state_tensor = (
            torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor).cpu().numpy()[0]
        self.qnetwork_local.train()

        # Mask invalid actions with large negative value
        invalid_mask = ~valid_actions
        action_values[invalid_mask] = -np.inf

        return np.argmax(action_values)

    def step(self, state, action, reward, next_state, done):
        """Store experience in replay buffer and learn if enough samples"""
        # Add experience to memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.target_update
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                self.learn()

    def learn(self):
        """Update value parameters using batch of experience tuples"""
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Convert to tensors and add channel dimension for CNN
        states = torch.from_numpy(states).float().unsqueeze(1).to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().unsqueeze(1).to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # Get Q values for next states from target model
        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get current Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss and update
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()
