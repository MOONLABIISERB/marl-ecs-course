import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import math
import wandb  # Add wandb import

# Define transition tuple structure
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class BoatNetwork(nn.Module):
    def __init__(self, grid_size, n_actions=8):
        super(BoatNetwork, self).__init__()

        # Process grid with CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Calculate CNN output size
        conv_out_size = 64 * grid_size * grid_size

        # Process boat positions
        self.fc_pos = nn.Linear(4, 64)  # 2 boats x 2 coordinates

        # Process other features
        self.fc_other = nn.Linear(
            4, 32
        )  # tether_length + steps_remaining + active_boat + trash_remaining

        # Combine all features
        self.fc_combine = nn.Linear(conv_out_size + 64 + 32, 512)
        self.fc_hidden = nn.Linear(512, 256)
        self.fc_advantage = nn.Linear(256, n_actions)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, state_dict):
        # Process grid
        grid = state_dict["grid"].float().unsqueeze(1)  # Add channel dimension
        grid_features = self.conv(grid)
        grid_features = grid_features.view(grid_features.size(0), -1)

        # Process boat positions
        pos = (
            state_dict["boat_positions"]
            .float()
            .view(state_dict["boat_positions"].size(0), -1)
        )
        pos_features = F.relu(self.fc_pos(pos))

        # print(state_dict)

        # print dimension of other features
        # print(state_dict["tether_length"].squeeze(-1).shape)
        # print(state_dict["steps_remaining"].squeeze(-1).shape)
        # print(state_dict["active_boat"].squeeze(-1).shape)
        # print(state_dict["trash_remaining"].squeeze(-1).shape)

        # Process other features
        other = torch.cat(
            [
                state_dict["tether_length"].squeeze(-1),
                state_dict["steps_remaining"].squeeze(-1),
                state_dict["active_boat"].float(),
                state_dict["trash_remaining"].squeeze(-1),
            ],
            dim=1,
        )
        other_features = F.relu(self.fc_other(other))

        # Combine all features
        combined = torch.cat([grid_features, pos_features, other_features], dim=1)
        hidden = F.relu(self.fc_combine(combined))
        hidden = F.relu(self.fc_hidden(hidden))

        # Dueling DQN architecture
        advantage = self.fc_advantage(hidden)
        value = self.fc_value(hidden)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TetheredBoatsAgent:
    def __init__(
        self, grid_size, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.grid_size = grid_size

        # Networks and optimizer
        self.policy_net = BoatNetwork(grid_size).to(device)
        self.target_net = BoatNetwork(grid_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters())

        # Replay memory
        self.memory = ReplayMemory(100000)

        # Training parameters
        self.batch_size = 256
        self.gamma = 0.95
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 20000
        self.target_update = 1000
        self.steps_done = 0

    def _to_tensor(self, state_dict):
        """Convert numpy state dict to tensor state dict"""
        return {
            "grid": torch.FloatTensor(state_dict["grid"]).unsqueeze(0).to(self.device),
            "boat_positions": torch.FloatTensor(state_dict["boat_positions"])
            .unsqueeze(0)
            .to(self.device),
            "tether_length": torch.FloatTensor(np.array([state_dict["tether_length"]]))
            .unsqueeze(0)
            .to(self.device),
            "steps_remaining": torch.FloatTensor(
                np.array([state_dict["steps_remaining"]])
            )
            .unsqueeze(0)
            .to(self.device),
            "active_boat": torch.FloatTensor([state_dict["active_boat"]])
            .unsqueeze(0)
            # .unsqueeze(-1)
            .to(self.device),
            "trash_remaining": torch.FloatTensor(
                np.array([state_dict["trash_remaining"]])
            )
            .unsqueeze(0)
            .to(self.device),
        }

    def select_action(self, state_dict, evaluate=False):
        """Select action for active boat"""
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )

        if not evaluate and random.random() < eps_threshold:
            return random.randint(0, 7)  # Random action

        with torch.no_grad():
            # print("State dict: ", state_dict)
            state_tensor = self._to_tensor(state_dict)
            # print("state tensor: ", state_tensor)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Create mask for non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        non_final_next_states = {
            "grid": torch.cat(
                [self._to_tensor(s)["grid"] for s in batch.next_state if s is not None]
            ),
            "boat_positions": torch.cat(
                [
                    self._to_tensor(s)["boat_positions"]
                    for s in batch.next_state
                    if s is not None
                ]
            ),
            "tether_length": torch.cat(
                [
                    self._to_tensor(s)["tether_length"]
                    for s in batch.next_state
                    if s is not None
                ]
            ),
            "steps_remaining": torch.cat(
                [
                    self._to_tensor(s)["steps_remaining"]
                    for s in batch.next_state
                    if s is not None
                ]
            ),
            "active_boat": torch.cat(
                [
                    self._to_tensor(s)["active_boat"]
                    for s in batch.next_state
                    if s is not None
                ]
            ),
            "trash_remaining": torch.cat(
                [
                    self._to_tensor(s)["trash_remaining"]
                    for s in batch.next_state
                    if s is not None
                ]
            ),
        }

        state_batch = {
            "grid": torch.cat([self._to_tensor(s)["grid"] for s in batch.state]),
            "boat_positions": torch.cat(
                [self._to_tensor(s)["boat_positions"] for s in batch.state]
            ),
            "tether_length": torch.cat(
                [self._to_tensor(s)["tether_length"] for s in batch.state]
            ),
            "steps_remaining": torch.cat(
                [self._to_tensor(s)["steps_remaining"] for s in batch.state]
            ),
            "active_boat": torch.cat(
                [self._to_tensor(s)["active_boat"] for s in batch.state]
            ),
            "trash_remaining": torch.cat(
                [self._to_tensor(s)["trash_remaining"] for s in batch.state]
            ),
        }

        action_batch = torch.tensor(batch.action, device=self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss and optimize
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def train_episode(self, env):
        """Train for one episode"""
        state = env.reset()
        # print(state)
        total_reward = 0
        losses = []

        # Add active_boat to state
        state["active_boat"] = torch.tensor([0])  # Start with boat 0

        while True:
            # print(state)
            # Get action for active boat
            action = self.select_action(state)

            # Create full action (one boat moves at a time)
            full_action = [8, 8]  # Stay action for inactive boat
            full_action[state["active_boat"]] = action

            # Take action in environment
            next_state, reward, done, _ = env.step(full_action)

            # Switch active boat
            next_state["active_boat"] = 1 - state["active_boat"]

            # Store transition
            self.memory.push(
                state, action, next_state if not done else None, reward, done
            )

            # Move to next state
            state = next_state
            total_reward += reward

            # Optimize model
            if len(self.memory) >= self.batch_size:
                loss = self.optimize_model()
                losses.append(loss)

            # Update target network
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.steps_done += 1

            if done:
                print("trash left: ", len(state["trash_positions"]))
                break

            # env.render()

            # trash_collected = len(state["trash_positions"])

        return (
            total_reward,
            np.mean(losses) if losses else 0,
            # trash_collected,
        )

    def evaluate_episode(self, env):
        """Evaluate for one episode"""
        state = env.reset()
        total_reward = 0
        state["active_boat"] = 0

        while True:
            # Get action for active boat
            action = self.select_action(state, evaluate=False)

            # Create full action
            full_action = [8, 8]
            full_action[state["active_boat"]] = action

            # Take action
            next_state, reward, done, _ = env.step(full_action)
            next_state["active_boat"] = 1 - state["active_boat"]

            state = next_state
            total_reward += reward

            if done:
                break

            env.render()
        # env.close()

        return total_reward

    def save(self, path):
        """Save model"""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["steps_done"]


# Training loop example
if __name__ == "__main__":
    import gym
    from env2 import TetheredBoatsEnv

    wandb.init(project="CleanSweepRL", name="random")  # Initialize wandb

    # Create environment and agent
    env = TetheredBoatsEnv(
        grid_size=10,
        n_boats=2,
        tether_length=4,
        time_penalty=-1,
        trash_left_penalty=0,
        trash_reward=0,
        complete_reward=10,
        incomplete_penalty=0,
        invalid_move_penalty=0,
        step_per_episode=150,
        num_episode=1,
        seed=None,
        n_trash=5,
    )
    agent = TetheredBoatsAgent(env.grid_size)

    # Training loop
    n_episodes = 1000
    best_reward = float("-inf")

    for episode in range(n_episodes):
        # Train episode
        # reward, loss = agent.train_episode(env)

        # # Assume env has an attribute 'trash_left'
        # # trash_left = env.trash_left  # Get trash left

        # # Save best model
        # if reward > best_reward:
        #     best_reward = reward
        #     agent.save("best_model.pth")

        # # Print progress
        # print(
        #     f"Episode {episode+1}/{n_episodes} - Reward: {reward:.2f}, Loss: {loss:.4f}"
        # )

        # # Log metrics to wandb
        # wandb.log(
        #     {
        #         "Episode": episode + 1,
        #         "Reward": reward,
        #         "Loss": loss,
        #         # "Trash Left": trash_left,
        #     }
        # )

        # Evaluate every 20 episodes
        if (episode + 1) % 5 == 0:
            eval_reward = agent.evaluate_episode(env)
            print(f"Evaluation reward: {eval_reward:.2f}")
            wandb.log({"Evaluation Reward": eval_reward})

    env.close()
