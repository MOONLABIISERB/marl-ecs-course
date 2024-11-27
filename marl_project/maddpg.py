import numpy as np
import torch
import torch.nn as nn
from torch import optim
from collections import deque
import random
import wandb
from rich.progress import track
from environment import MultiAgentShipTowEnv
from datetime import datetime


wandb.init(project="marl")  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# class StateNormalizer:
#     def __init__(self):
#         self.mean = None
#         self.std = None
#         self.count = 0
#         self.first_encountered = True

#     def __call__(self, x):
#         # Incrementally update mean and std
#         if self.first_encountered:
#             self.mean = np.zeros_like(x, dtype=np.float32)
#             self.std = np.zeros_like(x, dtype=np.float32)
#             self.first_encountered = False

#         # Update count
#         self.count += 1

#         # Incremental mean and variance update (Welford's algorithm)
#         delta = x - self.mean
#         self.mean += delta / self.count
#         delta2 = x - self.mean
#         self.std += delta * delta2

#         # Normalize
#         if self.count > 1:
#             variance = self.std / (self.count - 1)
#             std = np.sqrt(variance + 1e-8)  # Add small epsilon to prevent division by zero
#             return (x - self.mean) / std
        
#         return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        """
        Initialize the Actor network
        """
        super(Actor, self).__init__()
        self.act = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.act(x)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, n_agents=2) -> None:
        """
        Initialize the Critic network
        """
        super(Critic, self).__init__()

        self.crt = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.crt(x)


class MADDPGAgent:
    def __init__(self, state_size, action_size, hidden_size=128) -> None:
        """
        Initialize the MADDPG Agent
        """
        # self.state_normalizer = StateNormalizer()
    
        self.actor = Actor(state_size, hidden_size, action_size).to(device)
        self.critic = Critic(state_size, action_size, hidden_size).to(device)

        self.target_actor = Actor(state_size, hidden_size, action_size).to(device)
        self.target_critic = Critic(state_size, action_size, hidden_size).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.003)

        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.01

    def normalize_state(self, state):
        return self.state_normalizer(state)

    def update_targets(self, tau) -> None:
        """
        Update target networks

        Args:
            tau (float): parameter for soft update

        Returns:
            None
        """
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)



class MADDPG:
    def __init__(self, env) -> None:
        """
        Initialize the MADDPG
        """
        self.env = env
        self.n_agents = 2  
        
        state_size = env.observation_space['tugboat_1'].shape[0]
        action_size = env.action_space['tugboat_1'].shape[0]
        
        self.agents = [MADDPGAgent(state_size, action_size) for _ in range(self.n_agents)]

    def train(self, n_episodes, max_steps=1000) -> None:
        """
        Train the MADDPG agent

        Args:
            n_episodes (int): number of episodes to train
            max_steps (int): maximum number of steps per episode

        Returns:
            None
        """
        best_reward = -np.inf
        for episode in track(range(n_episodes), description="Episodes"):
            observations = self.env.reset()
            episode_reward = 0
            episode_reward_1 = 0
            episode_reward_2 = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                actions = {}
                for i, agent_id in enumerate(['tugboat_1', 'tugboat_2']):
                    # Normalize the state before passing to actor
                    # normalized_obs = self.agents[i].normalize_state(observations[agent_id])
                    # obs = torch.FloatTensor(normalized_obs).to(device)
                    obs = torch.FloatTensor(observations[agent_id]).to(device)
                    with torch.no_grad():
                        action = self.agents[i].actor(obs).cpu().numpy()
                        # exploration noise
                        action += np.random.normal(0, 0.1, size=action.shape)
                        action = np.clip(action, 0, 10)
                        actions[agent_id] = action
                
                next_observations, rewards, dones, _ = self.env.step(actions)
                done = dones['__all__']
                
                for i, agent_id in enumerate(['tugboat_1', 'tugboat_2']):
                    self.agents[i].memory.append((
                        observations[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_observations[agent_id],
                        done
                    ))
                
                observations = next_observations
                episode_reward_1 += rewards['tugboat_1']
                episode_reward_2 += rewards['tugboat_2']
                step += 1
                
                if len(self.agents[0].memory) > self.agents[0].batch_size:
                    self._update_agents()
                
                # self.env.render()
            
            wandb.log({
                "Episode Reward for tugboat 1": episode_reward_1,
                "Episode Reward for tugboat 2": episode_reward_2,
                "Episode Reward": episode_reward,
                "Episode": episode, 
                "Steps": step,
                "Best Reward": best_reward
            })

            if episode_reward > best_reward:
                best_reward = episode_reward
            
            if episode % 100 == 0 or episode == n_episodes - 1:
                self._save_models()

    def _update_agents(self) -> None:
        """
        Update the agents using the MADDPG algorithm

        Returns:
            None
        """
        for i, agent in enumerate(self.agents):
            batch = random.sample(agent.memory, agent.batch_size)
            states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
            
            # Normalize states
            # states = np.array([agent.normalize_state(s) for s in states])
            # next_states = np.array([agent.normalize_state(s) for s in next_states])
            
            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(-1).to(device)
            
            # Updating Critic
            next_actions = agent.target_actor(next_states)
            target_Q = rewards + agent.gamma * (1 - dones) * agent.target_critic(next_states, next_actions)
            current_Q = agent.critic(states, actions)
            critic_loss = nn.MSELoss()(current_Q, target_Q.detach())
            
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            wandb.log({f"Agent {i+1} Critic Loss": critic_loss.item()})
            
            # Updating Actor
            actor_loss = -agent.critic(states, agent.actor(states)).mean()
            
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            wandb.log({f"Agent {i+1} Actor Loss": actor_loss.item()})
            
            agent.update_targets(agent.tau)
    

    def _save_models(self) -> None:
        """
        Save the models to disk
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f'saved_models/tugboat_{i+1}_{timestamp}_actor_maddpg.pth')
            # torch.save(agent.critic.state_dict(), f'tugboat_{i+1}_{timestamp}_critic_maddpg.pth')


if __name__=='__main__':
    # import os
    # if not os.path.exists('saved_models'):
    #     os.makedirs('saved_models')

    env = MultiAgentShipTowEnv()
    maddpg = MADDPG(env)
    num_episodes = int(input("Number of Episodes: "))
    max_steps = int(input("Max Steps per Episode: "))
    maddpg.train(n_episodes=num_episodes, max_steps=max_steps)