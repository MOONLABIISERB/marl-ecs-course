import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from assignments.a3.env import GridWorldEnv
from assignments.a3.viz import run_episode_with_rendering


class MAPFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize with small weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 0.01)
                layer.bias.data.zero_()

    def forward(self, x):
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value

    def get_value(self, x):
        return self.critic(x)

    def evaluate_actions(self, x, actions):
        action_logits, values = self.forward(x)
        dist = Categorical(logits=action_logits)

        action_log_probs = dist.log_prob(actions.squeeze(-1))
        dist_entropy = dist.entropy()

        return action_log_probs, dist_entropy, values

    def get_action(self, x):
        action_logits, value = self.forward(x)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action, action_log_prob, value


class MAPFTrainer:
    def __init__(
        self,
        env,
        hidden_dim=64,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        n_epochs=4,
        batch_size=32,
        max_grad_norm=0.5,
    ):
        self.env = env
        self.n_agents = env.n_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create networks and optimizers for each agent
        self.networks = []
        self.optimizers = []

        for _ in range(self.n_agents):
            network = MAPFNetwork(
                input_dim=env.observation_space.shape[0],
                hidden_dim=hidden_dim,
                n_actions=env.action_space.n,
            ).to(self.device)

            optimizer = optim.Adam(network.parameters(), lr=lr, eps=1e-5)

            self.networks.append(network)
            self.optimizers.append(optimizer)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = deque(maxlen=100)

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = (
                rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            )
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )

        returns = advantages + values

        return returns, advantages

    def update_policy(
        self, agent_id, states, actions, old_log_probs, returns, advantages
    ):
        batch_size = len(states)

        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            # Generate random mini-batches
            indices = torch.randperm(batch_size)

            for start_idx in range(0, batch_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Get current policy outputs
                new_log_probs, entropy, values = self.networks[
                    agent_id
                ].evaluate_actions(batch_states, batch_actions)

                # Calculate policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Calculate value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Calculate total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                # Update network
                self.optimizers[agent_id].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.networks[agent_id].parameters(), self.max_grad_norm
                )
                self.optimizers[agent_id].step()

    def train(self, n_episodes=1000, max_steps=100):
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            step = 0

            states = [[] for _ in range(self.n_agents)]
            actions = [[] for _ in range(self.n_agents)]
            rewards = [[] for _ in range(self.n_agents)]
            values = [[] for _ in range(self.n_agents)]
            log_probs = [[] for _ in range(self.n_agents)]
            dones = [[] for _ in range(self.n_agents)]

            done = False
            while not done and step < max_steps:
                # Get actions for all agents
                actions_t = []

                for agent_id in range(self.n_agents):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action, log_prob, value = self.networks[agent_id].get_action(
                            state_tensor
                        )

                    states[agent_id].append(state)
                    actions_t.append(action.item())
                    values[agent_id].append(value.item())
                    log_probs[agent_id].append(log_prob.item())

                # Take environment step
                next_state, rewards_t, done, _, _ = self.env.step(actions_t)

                # Store experiences for each agent
                for agent_id in range(self.n_agents):
                    actions[agent_id].append(actions_t[agent_id])
                    rewards[agent_id].append(rewards_t[agent_id])
                    dones[agent_id].append(done)

                state = next_state
                episode_reward += sum(rewards_t)
                step += 1

            # Get final value estimate
            final_values = []
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            for agent_id in range(self.n_agents):
                with torch.no_grad():
                    final_value = (
                        self.networks[agent_id].get_value(state_tensor).cpu().item()
                    )
                final_values.append(final_value)

            # Update policy for each agent
            for agent_id in range(self.n_agents):
                if len(states[agent_id]) > 0:  # Only update if we have data
                    agent_values = torch.FloatTensor(values[agent_id])
                    agent_rewards = torch.FloatTensor(rewards[agent_id])
                    agent_dones = torch.FloatTensor(dones[agent_id])

                    # Compute returns and advantages
                    returns, advantages = self.compute_gae(
                        agent_rewards, agent_values, agent_dones, final_values[agent_id]
                    )

                    self.update_policy(
                        agent_id,
                        states[agent_id],
                        actions[agent_id],
                        log_probs[agent_id],
                        returns,
                        advantages,
                    )

            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step)
            self.success_rate.append(1 if step < max_steps else 0)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_success = np.mean(list(self.success_rate))
                print(f"Episode {episode + 1}")
                print(f"Average Reward (last 100): {avg_reward:.2f}")
                print(f"Average Length (last 100): {avg_length:.2f}")
                print(f"Success Rate (last 100): {avg_success:.2%}")
                print("----------------------------------------")

        return self.episode_rewards, self.episode_lengths


def plot_training_results(rewards, lengths):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True)

    # Plot episode lengths
    ax2.plot(lengths)
    ax2.set_title("Episode Lengths")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


walls = [
    [2, 2],
    [2, 3],
    [2, 4],  # Vertical wall
    [4, 4],
    [5, 4],
    [6, 4],  # Horizontal wall
]

goal_positions = [
    [7, 7],  # Goal for agent 1
    [7, 6],  # Goal for agent 2
    [6, 7],  # Goal for agent 3
]

env = GridWorldEnv(
    grid_size=8,
    n_agents=3,
    walls=walls,
    goal_positions=goal_positions,
    random_start=True,
)

trainer = MAPFTrainer(
    env,
    hidden_dim=64,
    lr=2.5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    n_epochs=4,
    batch_size=32,
)

rewards, lengths = trainer.train(n_episodes=1000, max_steps=100)
plot_training_results(rewards, lengths)
run_episode_with_rendering(env=env)
