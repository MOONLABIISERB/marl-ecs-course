import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from collections import namedtuple, deque
import gymnasium as gym
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on ", device)

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# class DQN(nn.Module):
#     def __init__(self, state_dim, n_actions):
#         super(DQN, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=state_dim, num_heads=4)
#         self.fc1 = nn.Linear(state_dim, 256)
#         self.fc2 = nn.Linear(256, n_actions)

#     def forward(self, x):
#         x, _ = self.attention(x, x, x)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)


class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc5(x)
        return x


class Agent:
    def __init__(self, memory_capacity: int, state_dim: int, n_actions: int, gamma: float):
        self.memory = ReplayMemory(memory_capacity)
        self.knowledge = DQN(state_dim, n_actions).to(device)
        self.target_net = DQN(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.knowledge.state_dict())
        self.target_net.eval()

        self.n_actions = n_actions
        self.gamma = gamma

        self.optimizer = optim.Adam(self.knowledge.parameters(), lr=0.0005)
        # self.loss_fn = F.smooth_l1_loss()

    def save_model(self, path: str):
        torch.save(self.knowledge.state_dict(), path)
        print(f"Model saved to {path}")

    def action(self, state: np.ndarray, exploration_prob: float):
        visited_states = state[-10:]

        # print(visited_states)
        unvisited_states = [i for i, s in enumerate(visited_states) if s == 0]

        if np.random.rand() < exploration_prob:
            action = random.choice(unvisited_states) if unvisited_states else np.random.randint(self.n_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.knowledge(state_tensor)

            # Mask Q-values of visited states to a very low value
            q_values_masked = q_values.clone()
            for idx, visited in enumerate(visited_states):
                if visited == 1:
                    q_values_masked[0, idx] = float("-inf")

            action = q_values_masked.argmax().item()

        return action

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn_from_memory(self, batch_size: int):
        if batch_size > len(self.memory):
            return None

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done_batch = torch.FloatTensor(batch.done).to(device)

        state_action_values = self.knowledge(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]

        expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.knowledge.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_knowledge(self):
        self.target_net.load_state_dict(self.knowledge.state_dict())
