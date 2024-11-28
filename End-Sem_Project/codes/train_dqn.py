import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from env2 import CaptureTheFlagEnv 


class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.2, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = []
        self.batch_size = 64
        self.max_buffer_size = 10000

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def update_replay_buffer(self, experience):
        """Store experience in replay buffer."""
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(experience)

    def train(self):
        """Train the Q-network using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = nn.HuberLoss(reduction='mean')(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def update_target_network(self):
        """Update target network to match Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn(env, num_episodes=1000, max_steps_per_episode=200, target_update_freq=10, log_dir="logs"):
    """Train agents using DQN with logging and saving."""
    agents = {agent_id: DQNAgent(state_dim=env.grid_size[0] * env.grid_size[1], action_dim=5) for agent_id in env.positions.keys()}
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{log_dir}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    plt.ion()
    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    ax[0].set_title("Agent Movements")
    ax[1].set_title("Episode Rewards")
    ax[1].set_xlabel("Episodes")
    ax[1].set_ylabel("Total Rewards")
    ax[2].set_title("Episode Loss")
    ax[2].set_xlabel("Episodes")
    ax[2].set_ylabel("Average Loss")

    reward_history_team_1 = []
    reward_history_team_2 = []
    average_loss_history = []
    logs = {"episode_rewards": [], "team_scores": [], "losses": []}

    for episode in range(num_episodes):
        _, state = env.reset()
        state = {agent_id: np.array(state[agent_id]).flatten() for agent_id in state}
        episode_rewards = {"team_1": 0, "team_2": 0}
        episode_losses = []

        for step in range(max_steps_per_episode):
            actions = {agent_id: agent.choose_action(state[agent_id]) for agent_id, agent in agents.items()}
            next_state, rewards, done, scores = env.step(actions)
            next_state = {agent_id: np.array(next_state[agent_id]).flatten() for agent_id in next_state}

            for agent_id, agent in agents.items():
                done_flag = 1 if done["__all__"] else 0
                agent.update_replay_buffer((state[agent_id], actions[agent_id], rewards[agent_id], next_state[agent_id], done_flag))
                loss = agent.train()
                if loss is not None:  
                    episode_losses.append(loss.item())

                if agent_id.startswith("team_1"):
                    episode_rewards["team_1"] += rewards[agent_id]
                elif agent_id.startswith("team_2"):
                    episode_rewards["team_2"] += rewards[agent_id]

            state = next_state
            visualize_movement(env, ax[0], step, episode, scores)
            plt.pause(0.1)

            if done["__all__"]:
                break

        if episode % target_update_freq == 0:
            for agent in agents.values():
                agent.update_target_network()

        reward_history_team_1.append(episode_rewards["team_1"])
        reward_history_team_2.append(episode_rewards["team_2"])
        average_loss = np.mean(episode_losses) if episode_losses else 0
        average_loss_history.append(average_loss)
        logs["episode_rewards"].append(episode_rewards)
        logs["team_scores"].append(scores)
        logs["losses"].append(average_loss)

        visualize_rewards(ax[1], reward_history_team_1, reward_history_team_2, episode + 1)
        visualize_loss(ax[2], average_loss_history, episode + 1)

        if (episode + 1) % 50 == 0:
            save_agents(agents, f"{log_dir}/dqn_agents_ep{episode + 1}.pkl")

        print(f"Episode {episode + 1}/{num_episodes}: Team 1 Rewards = {episode_rewards['team_1']}, Team 2 Rewards = {episode_rewards['team_2']}, Average Loss = {average_loss:.4f}")

    save_agents(agents, f"{log_dir}/final_dqn_agents.pkl")
    save_logs(logs, f"{log_dir}/training_logs.pkl")
    plt.ioff()
    plt.show()


def visualize_loss(ax, loss_history, episode):
    """Update the loss plot in real-time."""
    ax.clear()
    ax.plot(range(1, episode + 1), loss_history, label="Average Loss", color="orange")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    ax.set_title("Episode Loss")
    ax.legend()
def visualize_movement(env, ax, step, episode, scores):
    """Render the current state of the environment in real-time."""
    ax.clear()
    ax.set_xticks(np.arange(0, env.grid_size[1] + 1, 1))
    ax.set_yticks(np.arange(0, env.grid_size[0] + 1, 1))
    ax.grid(True)
    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle((obs[1], obs[0]), 1, 1, color="grey"))
    for agent, pos in env.positions.items():
        color = "cyan" if "team_1" in agent else "green"
        ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, color=color))
        ax.text(pos[1] + 0.5, pos[0] + 0.5, agent[-1], ha="center", va="center", color="black")
    for flag, pos in env.flags.items():
        color = "blue" if "team_1" in flag else "red"
        ax.add_patch(plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.4, color=color))

    ax.set_xlim(0, env.grid_size[1])
    ax.set_ylim(0, env.grid_size[0])
    ax.set_aspect("equal")
    ax.set_title(f"Episode {episode + 1}, Step {step + 1} | Scores: {scores}")
def visualize_rewards(ax, team_1_rewards, team_2_rewards, episode):
    """Update the rewards plot in real-time."""
    ax.clear()
    ax.plot(range(1, episode + 1), team_1_rewards, label="Team 1 Rewards", color="cyan")
    ax.plot(range(1, episode + 1), team_2_rewards, label="Team 2 Rewards", color="green")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Rewards")
    ax.set_title("Episode Rewards")
    ax.legend()
def save_agents(agents, filename):
    """Save all agent models to a file."""
    agent_data = {agent_id: agent.q_network.state_dict() for agent_id, agent in agents.items()}
    with open(filename, "wb") as f:
        pickle.dump(agent_data, f)
    print(f"Agents saved to {filename}")


def save_logs(logs, filename):
    """Save training logs to a file."""
    with open(filename, "wb") as f:
        pickle.dump(logs, f)
    print(f"Logs saved to {filename}")


if __name__ == "__main__":
    import os

    grid_size = (10, 10)
    team_size = 2
    env = CaptureTheFlagEnv(grid_size=grid_size, team_size=team_size)

    train_dqn(env, num_episodes=10000, max_steps_per_episode=150, log_dir="logs")
