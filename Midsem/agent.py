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


class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Agent:
    def __init__(self, memory_capacity: int, state_dim: int, n_actions: int, gamma: float):
        self.memory = ReplayMemory(memory_capacity)
        self.knowledge = DQN(state_dim, n_actions).to(device)
        self.target_net = DQN(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.knowledge.state_dict())
        self.target_net.eval()

        self.n_actions = n_actions
        self.gamma = gamma

        self.optimizer = optim.Adam(self.knowledge.parameters(), lr=0.0001)
        self.loss_fn = nn.SmoothL1Loss()

    def action(self, state: np.ndarray, exploration_prob: float):
        if np.random.rand() < exploration_prob:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.knowledge(state)
            return q_values.argmax().item()

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

        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.knowledge.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_knowledge(self):
        self.target_net.load_state_dict(self.knowledge.state_dict())


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="dqn-cartpole", name="DQN-CartPole-v1")

    # Usage example
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(memory_capacity=10000, state_dim=state_dim, n_actions=n_actions, gamma=0.99)
    batchSize = 512

    # Training loop (modified to use wandb)
    num_episodes = 1000
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_losses = []

        while not done:
            action = agent.action(state, exploration_prob=0.05)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update_memory(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Learn and track loss
            if len(agent.memory) >= batchSize:
                loss = agent.learn_from_memory(batch_size=batchSize)
                if loss is not None:
                    episode_losses.append(loss)

        if episode_losses:
            avg_loss = np.mean(episode_losses)
            wandb.log({"episode": episode, "avg_loss": avg_loss, "episode_reward": episode_reward})
        else:
            avg_loss = 0

        if episode % 10 == 0:
            agent.update_knowledge()
            print(f"Episode {episode}, Avg Loss: {avg_loss:.4f}, Reward: {episode_reward}")

    wandb.finish()
