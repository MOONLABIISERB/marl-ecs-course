import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from typing import Dict, List, Tuple
import os

class DiscreteActorNetwork(nn.Module):
    """Decentralized Actor Network for Discrete MADDPG"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)  # Softmax for discrete action probabilities

class DiscreteCriticNetwork(nn.Module):
    """Centralized Critic Network for Discrete MADDPG"""
    def __init__(self, state_dim, action_dim, n_agents):
        super().__init__()
        total_state_dim = state_dim * n_agents
        total_action_dim = action_dim * n_agents

        self.fc1 = nn.Linear(total_state_dim + total_action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, states, actions_one_hot):
        x = torch.cat([states, actions_one_hot], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DiscreteMAPPG:
    def __init__(self, 
                 state_dim,
                 action_dim, 
                 n_agents, 
                 lr_actor=1e-4, 
                 lr_critic=1e-3,
                 gamma=0.99,
                 tau=0.001,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=0.995):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Actor and Critic Networks for each agent
        self.actors = [DiscreteActorNetwork(state_dim, action_dim) for _ in range(n_agents)]
        self.target_actors = [DiscreteActorNetwork(state_dim, action_dim) for _ in range(n_agents)]
        
        # Centralized Critic for entire system
        self.critic = DiscreteCriticNetwork(state_dim, action_dim, n_agents)
        self.target_critic = DiscreteCriticNetwork(state_dim, action_dim, n_agents)

        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Weight initialization and target network initialization
        self._init_networks()

    def _init_networks(self):
        """Initialize network weights and copy weights to target networks"""
        for i in range(self.n_agents):
            self._soft_update(self.actors[i], self.target_actors[i], 1.0)
        self._soft_update(self.critic, self.target_critic, 1.0)

    def _soft_update(self, local_model, target_model, tau):
        """Soft update of target network weights"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def act(self, observations, train_mode=True):
        """Generate actions for all agents with epsilon-greedy exploration"""
        actions = []
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            if train_mode and random.random() < self.epsilon:
                # Random action for exploration
                action = random.randint(0, self.action_dim - 1)
            else:
                # Policy-based action selection
                with torch.no_grad():
                    action_probs = self.actors[i](obs_tensor).squeeze(0)
                    action = torch.argmax(action_probs).item()
            
            actions.append(action)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return actions

    def train(self, memory):
        """Train the Discrete MADDPG algorithm"""
        if len(memory) < self.n_agents:  # Batch size should match number of agents
            return

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = memory.sample(self.n_agents)

        # Prepare tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # Prepare one-hot actions for critic
        actions_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
        
        # Compute target actions and Q values
        target_actions_probs = []
        target_actions_one_hot = []
        with torch.no_grad():
            for i in range(self.n_agents):
                target_action_prob = self.target_actors[i](next_states[:, i, :])
                target_action = torch.argmax(target_action_prob, dim=1)
                target_action_one_hot = F.one_hot(target_action, num_classes=self.action_dim).float()
                
                target_actions_probs.append(target_action_prob)
                target_actions_one_hot.append(target_action_one_hot)
            
            # Flatten target actions for critic
            target_actions_one_hot = torch.cat(target_actions_one_hot, dim=1)
            
            # Compute target Q values
            target_q_values = self.target_critic(
                next_states.view(next_states.size(0), -1),
                target_actions_one_hot
            )
            
            # Bellman equation
            y = rewards + (1 - dones) * self.gamma * target_q_values

        # Critic update
        current_q_values = self.critic(
            states.view(states.size(0), -1),
            actions_one_hot.view(actions_one_hot.size(0), -1)
        )
        critic_loss = F.mse_loss(current_q_values, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (decentralized)
        for i in range(self.n_agents):
            # Compute actor loss using centralized critic
            actor_actions_probs = [self.actors[j](states[:, j, :]) for j in range(self.n_agents)]
            actor_actions_one_hot = [F.one_hot(torch.argmax(probs, dim=1), num_classes=self.action_dim).float() 
                                     for probs in actor_actions_probs]
            actor_actions_one_hot = torch.cat(actor_actions_one_hot, dim=1)
            
            # Compute policy loss
            actor_q_values = self.critic(states.view(states.size(0), -1), actor_actions_one_hot)
            actor_loss = -actor_q_values.mean()
            
            # Add entropy regularization to encourage exploration
            entropy_loss = torch.mean(torch.sum(actor_actions_probs[i] * torch.log(actor_actions_probs[i] + 1e-10), dim=1))
            total_actor_loss = actor_loss - 0.01 * entropy_loss
            
            self.actor_optimizers[i].zero_grad()
            total_actor_loss.backward()
            self.actor_optimizers[i].step()

        # Soft update of target networks
        for i in range(self.n_agents):
            self._soft_update(self.actors[i], self.target_actors[i], self.tau)
        
        self._soft_update(self.critic, self.target_critic, self.tau)

        return critic_loss.item()

class ReplayBuffer:
    """Replay buffer for storing and sampling experiences"""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

def train_discrete_maddpg(
    env, 
    maddpg_agent, 
    n_episodes=1000, 
    max_steps=500, 
    save_dir='checkpoints', 
    save_frequency=100
):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    replay_buffer = ReplayBuffer()
    
    for episode in range(n_episodes):
        observations = env.reset()
        obs_list = list(observations.values())
        
        episode_rewards = {agent_id: 0 for agent_id in env.agents}
        
        for step in range(max_steps):
            # Get actions from policy
            actions = maddpg_agent.act(obs_list)
            action_dict = {agent_id: action for agent_id, action in zip(env.agents.keys(), actions)}
            
            # Step environment
            observations, rewards, dones, _ = env.step(action_dict)
            obs_list = list(observations.values())
            
            # Update rewards
            for agent_id in env.agents:
                episode_rewards[agent_id] += rewards[agent_id]
            
            # Store experiences for each agent
            for i, agent_id in enumerate(env.agents):
                replay_buffer.push(
                    state=obs_list[i],
                    action=actions[i],
                    reward=rewards[agent_id],
                    next_state=obs_list[i],
                    done=dones[agent_id]
                )
            
            # Train agent
            if len(replay_buffer) >= maddpg_agent.n_agents:
                critic_loss = maddpg_agent.train(replay_buffer)
            
            # Check if episode is done
            if all(dones.values()):
                break
        
        # Print episode statistics
        print(f"Episode {episode}: Total Rewards = {episode_rewards}, Epsilon = {maddpg_agent.epsilon:.4f}")
        
        # Save checkpoints periodically
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = os.path.join(save_dir, f'maddpg_checkpoint_episode_{episode+1}.pth')
            torch.save({
                'episode': episode,
                'maddpg_state_dict': {
                    'actors': [actor.state_dict() for actor in maddpg_agent.actors],
                    'target_actors': [target_actor.state_dict() for target_actor in maddpg_agent.target_actors],
                    'critic': maddpg_agent.critic.state_dict(),
                    'target_critic': maddpg_agent.target_critic.state_dict(),
                    'actor_optimizers': [optimizer.state_dict() for optimizer in maddpg_agent.actor_optimizers],
                    'critic_optimizer': maddpg_agent.critic_optimizer.state_dict(),
                    'epsilon': maddpg_agent.epsilon
                }
            }, checkpoint_path)
            print(f"Checkpoint saved at episode {episode+1}")

    return maddpg_agent

# Uncomment and modify the following lines based on your environment setup
# discrete_maddpg = DiscreteMAPPG(
#     state_dim=env.observation_space[0].shape[0],
#     action_dim=4,  # Number of possible actions
#     n_agents=len(env.agents)
# )
# trained_agent = train_discrete_maddpg(
#     env, 
#     discrete_maddpg, 
#     n_episodes=1000, 
#     max_steps=1000, 
#     save_dir='checkpoints', 
#     save_frequency=100
# )