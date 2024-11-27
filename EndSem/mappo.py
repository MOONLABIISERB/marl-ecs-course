import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class CNNEncoder(nn.Module):
    def __init__(self, input_channels=4):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)  # Flatten


class Actor(nn.Module):
    def __init__(self, grid_size=10, action_dim=9):
        super(Actor, self).__init__()

        self.cnn = CNNEncoder()
        cnn_output = 32 * grid_size * grid_size

        self.fc1 = nn.Linear(cnn_output + 4, 256)  # +4 for own_pos and other_pos
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, action_dim)

    def forward(self, grid_state, own_pos, other_pos, valid_actions):
        # Convert inputs to correct type and shape
        if not isinstance(grid_state, torch.Tensor):
            grid_state = torch.FloatTensor(grid_state)
        if not isinstance(own_pos, torch.Tensor):
            own_pos = torch.FloatTensor(own_pos)
        if not isinstance(other_pos, torch.Tensor):
            other_pos = torch.FloatTensor(other_pos)
        if not isinstance(valid_actions, torch.Tensor):
            valid_actions = torch.FloatTensor(valid_actions)

        # Ensure correct dimensions
        if len(grid_state.shape) == 3:
            grid_state = grid_state.unsqueeze(0)
        if len(own_pos.shape) == 1:
            own_pos = own_pos.unsqueeze(0)
        if len(other_pos.shape) == 1:
            other_pos = other_pos.unsqueeze(0)
        if len(valid_actions.shape) == 1:
            valid_actions = valid_actions.unsqueeze(0)

        # Process through networks
        x = self.cnn(grid_state)
        pos_features = torch.cat([own_pos, other_pos], dim=-1)
        x = torch.cat([x, pos_features], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_probs = F.softmax(self.policy_head(x), dim=-1)
        action_probs = action_probs * valid_actions
        action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-10)

        return action_probs


class Critic(nn.Module):
    def __init__(self, grid_size=10):
        super(Critic, self).__init__()

        self.cnn = CNNEncoder()
        cnn_output = 32 * grid_size * grid_size

        self.fc1 = nn.Linear(
            cnn_output + 8 + 18, 256
        )  # +8 for positions, +18 for actions
        self.fc2 = nn.Linear(256, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, grid_state, pos1, pos2, action1, action2):
        # Convert inputs to correct type and shape
        if not isinstance(grid_state, torch.Tensor):
            grid_state = torch.FloatTensor(grid_state)
        if not isinstance(pos1, torch.Tensor):
            pos1 = torch.FloatTensor(pos1)
        if not isinstance(pos2, torch.Tensor):
            pos2 = torch.FloatTensor(pos2)
        if not isinstance(action1, torch.Tensor):
            action1 = torch.FloatTensor(action1)
        if not isinstance(action2, torch.Tensor):
            action2 = torch.FloatTensor(action2)

        # Ensure correct dimensions
        if len(grid_state.shape) == 3:
            grid_state = grid_state.unsqueeze(0)
        if len(pos1.shape) == 1:
            pos1 = pos1.unsqueeze(0)
        if len(pos2.shape) == 1:
            pos2 = pos2.unsqueeze(0)

        x = self.cnn(grid_state)
        features = torch.cat([x, pos1, pos2, action1, action2], dim=-1)

        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)

        return value


class Memory:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.valid_actions = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def push(
        self, state, action, reward, next_state, valid_actions, log_prob, value, done
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.valid_actions.append(valid_actions)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)


class MAPPO:
    def __init__(
        self,
        env,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=10,
        batch_size=32,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs

        self.actor1 = Actor()
        self.actor2 = Actor()
        self.critic = Critic()

        self.actor1_opt = optim.Adam(self.actor1.parameters(), lr=lr_actor)
        self.actor2_opt = optim.Adam(self.actor2.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = Memory(batch_size)

    def _process_state(self, state):
        # Convert grid to one-hot encoding
        grid = torch.tensor(state["grid"], dtype=torch.int64)
        grid = F.one_hot(grid, num_classes=4).float()
        grid = grid.permute(2, 0, 1)  # [C, H, W]

        pos1 = torch.FloatTensor(state["pos1"])
        pos2 = torch.FloatTensor(state["pos2"])

        return grid, pos1, pos2

    def select_action(self, state, agent_id, valid_actions):
        grid, pos1, pos2 = self._process_state(state)
        valid_actions = torch.FloatTensor(valid_actions)

        with torch.no_grad():
            if agent_id == 0:
                probs = self.actor1(grid, pos1, pos2, valid_actions)
            else:
                probs = self.actor2(grid, pos2, pos1, valid_actions)

            # Create distribution and sample
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def update(self):
        batch = self.memory

        # Process all data into tensors
        states = [self._process_state(s) for s in batch.states]
        grids = torch.stack([s[0] for s in states])
        pos1s = torch.stack([s[1] for s in states])
        pos2s = torch.stack([s[2] for s in states])

        actions = torch.tensor(batch.actions)
        rewards = torch.tensor(batch.rewards)
        values = torch.tensor(batch.values)
        log_probs = torch.tensor(batch.log_probs)
        dones = torch.tensor(batch.dones)

        # Compute advantages and returns
        advantages = []
        returns = []
        next_value = 0
        next_advantage = 0

        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            next_value = 0 if d else next_value
            next_advantage = 0 if d else next_advantage

            delta = r + self.gamma * next_value - v
            advantage = delta + self.gamma * self.gae_lambda * next_advantage

            advantages.insert(0, advantage)
            returns.insert(0, advantage + v)

            next_value = v
            next_advantage = advantage

        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy
        for _ in range(self.epochs):
            # Update actor 1
            curr_probs1 = self.actor1(grids, pos1s, pos2s, batch.valid_actions[0])
            curr_dist1 = torch.distributions.Categorical(curr_probs1)
            curr_log_probs1 = curr_dist1.log_prob(actions[:, 0])

            ratio1 = torch.exp(curr_log_probs1 - log_probs[:, 0])
            surr1 = ratio1 * advantages
            surr2 = (
                torch.clamp(ratio1, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            )
            actor1_loss = -torch.min(surr1, surr2).mean()

            self.actor1_opt.zero_grad()
            actor1_loss.backward()
            self.actor1_opt.step()

            # Update actor 2
            curr_probs2 = self.actor2(grids, pos2s, pos1s, batch.valid_actions[1])
            curr_dist2 = torch.distributions.Categorical(curr_probs2)
            curr_log_probs2 = curr_dist2.log_prob(actions[:, 1])

            ratio2 = torch.exp(curr_log_probs2 - log_probs[:, 1])
            surr1 = ratio2 * advantages
            surr2 = (
                torch.clamp(ratio2, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            )
            actor2_loss = -torch.min(surr1, surr2).mean()

            self.actor2_opt.zero_grad()
            actor2_loss.backward()
            self.actor2_opt.step()

            # Update critic
            curr_values = self.critic(
                grids,
                pos1s,
                pos2s,
                F.one_hot(actions[:, 0], 9),
                F.one_hot(actions[:, 1], 9),
            )
            critic_loss = F.mse_loss(curr_values.squeeze(), returns)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        self.memory.reset()


def test_agent(env, mappo, num_episodes=10):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            valid_actions = state["valid_actions"]

            # Get actions from both agents
            action1, _ = mappo.select_action(state, 0, valid_actions[0])
            intermediate_state = env.step_agent(0, action1)
            action2, _ = mappo.select_action(intermediate_state, 1, valid_actions[1])

            # Take step in environment
            state, reward, done, _ = env.step([action1, action2])
            total_reward += reward

            env.render()

        print(f"Test Episode {episode}, Total Reward: {total_reward}")
