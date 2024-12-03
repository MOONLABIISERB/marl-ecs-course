import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import wandb
from rich.progress import track
from environment import MultiAgentShipTowEnv
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Initialize wandb for logging
wandb.init(project="marl-mappo")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),  # Added layer normalization
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # Added layer normalization
            nn.ReLU()
        )
        
        # Initialize mu and log_std with small weights
        self.mu = nn.Linear(hidden_size, action_size)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)
        
        self.log_std = nn.Parameter(torch.zeros(action_size))
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),  # Added layer normalization
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # Added layer normalization
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        # Orthogonal initialization for linear layers
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def get_action(self, state):
        # Forward through actor network
        x = self.actor(state)
        
        # Compute mean 
        mu = self.mu(x)
        
        # Use learned log_std with clipping
        log_std = torch.clamp(self.log_std, min=-5, max=2)
        std = torch.exp(log_std)
        
        # Create normal distribution
        dist = Normal(mu, std)
        
        # Sample action with reparameterization trick
        action = dist.rsample()
        
        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip action to valid range
        action = torch.clamp(action, 0, 50)
        
        return action, log_prob
    
    def get_value(self, state):
        return self.critic(state)

class MAPPOAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.actor_critic = ActorCritic(state_size, action_size, hidden_size).to(device)
        self.optimizer = optim.Adam(
            [
                {'params': self.actor_critic.actor.parameters()},
                {'params': self.actor_critic.mu.parameters()},
                {'params': self.actor_critic.log_std, 'lr': 1e-3},
                {'params': self.actor_critic.critic.parameters()}
            ], 
            lr=0.0003
        )
        
        # Hyperparameters
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.lmbda = 0.95
    
    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        advantages = []
        gae = 0
        
        # Compute advantages and returns in reverse order
        for i in reversed(range(len(rewards))):
            # Check if we're at the last step
            delta = rewards[i] + self.gamma * values[i+1] * (1-dones[i]) - values[i]
            gae = delta + self.gamma * self.lmbda * (1-dones[i]) * gae
            advantages.insert(0, gae)
        
        # Convert to tensor
        return torch.tensor(advantages, dtype=torch.float32).to(device)
    
    def update(self, states, actions, old_log_probs, rewards, dones):
        """
        PPO update step
        """
        # Convert states to tensor efficiently
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        
        # Compute returns and advantages
        values = []
        with torch.no_grad():
            for state in states_tensor:
                values.append(self.actor_critic.get_value(state))
            values.append(torch.tensor(0.0).to(device))  # Add a final value
        
        # Compute Generalized Advantage Estimation
        advantages = self.compute_gae(rewards, values, dones)
        
        # Compute returns
        returns = advantages + torch.stack(values[:-1]).detach()
        
        # Convert inputs to tensors
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        old_log_probs_tensor = torch.tensor(np.array(old_log_probs), dtype=torch.float32).to(device)
        
        # Optimize policy
        for _ in range(3):  # Multiple epochs of optimization
            # New policy
            new_actions, new_log_probs = self.actor_critic.get_action(states_tensor)
            
            # Policy ratio
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # Surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss (ensure correct dimensions)
            new_values = self.actor_critic.get_value(states_tensor).squeeze()
            value_loss = nn.MSELoss()(new_values, returns.detach())
            
            # Entropy loss to encourage exploration
            entropy_loss = -torch.mean(new_log_probs)
            
            # Total loss
            loss = (policy_loss + 
                    self.value_loss_coef * value_loss - 
                    self.entropy_coef * entropy_loss)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Log losses
            wandb.log({
                "Policy Loss": policy_loss.item(),
                "Value Loss": value_loss.item(),
                "Entropy Loss": entropy_loss.item()
            })

class MAPPO:
    def __init__(self, env):
        self.env = env
        self.n_agents = 2
        
        # Get state and action sizes
        state_size = env.observation_space['tugboat_1'].shape[0]
        action_size = env.action_space['tugboat_1'].shape[0]
        
        # Create agents
        self.agents = [MAPPOAgent(state_size, action_size) for _ in range(self.n_agents)]
    
    def train(self, n_episodes, max_steps=1000):
        best_reward = -np.inf
        
        for episode in track(range(n_episodes), description="Training Episodes"):
            observations = self.env.reset()
            episode_reward = 0
            episode_reward_1 = 0
            episode_reward_2 = 0
            done = False
            step = 0
            
            episode_memories = [[], []]
            
            while not done and step < max_steps:
                actions = {}
                log_probs = {}
                
                for i, agent_id in enumerate(['tugboat_1', 'tugboat_2']):
                    # Convert observation to tensor
                    obs = torch.FloatTensor(observations[agent_id]).to(device)
                    
                    # Get action and log probability
                    with torch.no_grad():
                        action, log_prob = self.agents[i].actor_critic.get_action(obs)
                    
                    # Convert to numpy for environment
                    actions[agent_id] = action.cpu().numpy()
                    log_probs[agent_id] = log_prob.cpu().numpy()
                
                # Take step in environment
                next_observations, rewards, dones, _ = self.env.step(actions)
                done = dones['__all__']
                
                # Store experiences for each agent
                for i, agent_id in enumerate(['tugboat_1', 'tugboat_2']):
                    episode_memories[i].append({
                        'state': observations[agent_id],
                        'action': actions[agent_id],
                        'reward': rewards[agent_id],
                        'done': dones[agent_id],
                        'log_prob': log_probs[agent_id]
                    })
                
                observations = next_observations
                episode_reward_1 += rewards['tugboat_1']
                episode_reward_2 += rewards['tugboat_2']
                step += 1
            
            # Process and update for each agent
            for i in range(self.n_agents):
                # Prepare data for update
                states = [mem['state'] for mem in episode_memories[i]]
                actions = [mem['action'] for mem in episode_memories[i]]
                rewards = [mem['reward'] for mem in episode_memories[i]]
                dones = [mem['done'] for mem in episode_memories[i]]
                old_log_probs = [mem['log_prob'] for mem in episode_memories[i]]
                
                # Update agent
                self.agents[i].update(states, actions, old_log_probs, rewards, dones)
            
            # Log episode metrics
            wandb.log({
                "Episode Reward for tugboat 1": episode_reward_1,
                "Episode Reward for tugboat 2": episode_reward_2,
                "Episode": episode, 
                "Steps": step,
                "Best Reward": best_reward
            })
            
            # Track best reward
            episode_reward = episode_reward_1 + episode_reward_2
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            # Save models periodically
            if episode % 100 == 0 or episode == n_episodes - 1:
                self._save_models()
    
    def _save_models(self):
        """Save the models for each agent"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, agent in enumerate(self.agents):
            torch.save(
                agent.actor_critic.state_dict(), 
                f'saved_models/tugboat_{i+1}_{timestamp}_mappo_actor_critic.pth'
            )

if __name__ == '__main__':
    # Create environment
    env = MultiAgentShipTowEnv()
    
    # Initialize MAPPO
    mappo = MAPPO(env)
    
    # Get training parameters
    num_episodes = int(input("Number of Episodes: "))
    max_steps = int(input("Max Steps per Episode: "))
    
    # Train
    mappo.train(n_episodes=num_episodes, max_steps=max_steps)
