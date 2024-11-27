import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from environment import MultiCarRacing
import random
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import json
import pygame

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experience replay memory tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ).to(device)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays first, then to tensors
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        # Convert numpy arrays to tensors and move to GPU in one operation
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class MADQN:
    def __init__(self, n_agents,
                 state_dim,
                 action_dim,
                 writer,
                 epsilon = 1.0,
                 epsilon_min = 0.1,
                 epsilon_decay = 0.5,
                 gamma = 0.95,
                 batch_size = 64,
                 target_update_frequency = 10
                 ):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.writer = writer
        
        # Create networks and replay buffers for each agent
        self.q_networks = {}
        self.target_networks = {}
        self.optimizers = {}
        self.replay_buffers = {}
        
        for agent_id in range(n_agents):
            self.q_networks[agent_id] = DQNetwork(state_dim, action_dim)
            self.target_networks[agent_id] = DQNetwork(state_dim, action_dim)
            self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())
            self.optimizers[agent_id] = optim.Adam(self.q_networks[agent_id].parameters())
            self.replay_buffers[agent_id] = ReplayBuffer(capacity=10000)
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.update_counter = 0
        self.training_step = 0
    
    def select_action(self, states):
        actions = {}
        for agent_id in range(self.n_agents):
            if random.random() < self.epsilon:
                actions[agent_id] = random.randint(0, self.action_dim - 1)
            else:
                state = torch.FloatTensor(states[agent_id]).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = self.q_networks[agent_id](state)
                    actions[agent_id] = q_values.argmax().item()
        return actions
    
    def update(self, agent_id):
        if len(self.replay_buffers[agent_id]) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffers[agent_id].sample(self.batch_size)
        
        # Compute current Q values
        current_q_values = self.q_networks[agent_id](states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_networks[agent_id](next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Log loss to TensorBoard
        self.writer.add_scalar(f'Loss/agent_{agent_id}', loss.item(), self.training_step)
        self.writer.add_scalar(f'Q_values/agent_{agent_id}', current_q_values.mean().item(), self.training_step)
        
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_step += 1

    def save_model(self, save_dir, episode):
        """Save model checkpoints and training state"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save model checkpoints for each agent
        checkpoint = {
            'episode': episode,
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'models': {},
            'optimizers': {}
        }
        
        for agent_id in range(self.n_agents):
            # Save Q-network state
            checkpoint['models'][f'agent_{agent_id}'] = self.q_networks[agent_id].state_dict()
            # Save optimizer state
            checkpoint['optimizers'][f'agent_{agent_id}'] = self.optimizers[agent_id].state_dict()
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint_episode_{episode}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save training configuration
        config = {
            'n_agents': self.n_agents,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update_frequency': self.target_update_frequency
        }
        
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def load_model(self, checkpoint_path):
        """Load model checkpoints and training state"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model parameters and optimizer states
        for agent_id in range(self.n_agents):
            self.q_networks[agent_id].load_state_dict(checkpoint['models'][f'agent_{agent_id}'])
            self.target_networks[agent_id].load_state_dict(checkpoint['models'][f'agent_{agent_id}'])
            self.optimizers[agent_id].load_state_dict(checkpoint['optimizers'][f'agent_{agent_id}'])
        
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        
        print(f"Loaded checkpoint from episode {checkpoint['episode']}")
        return checkpoint['episode']

def train_madqn(max_steps = 1000, n_episodes = 1000, save_dir='checkpoints', save_frequency=100):
    # Create unique run name with timestamp
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'MADQN_{current_time}')
    writer = SummaryWriter(log_dir)
    
    n_cars = 4
    env = MultiCarRacing(n_cars=n_cars, grid_size=30, track_width=5, render_mode=None)
    n_agents = 2
    state_dim = 30 * 30 + n_cars
    action_dim = 5
    
    madqn = MADQN(n_agents, state_dim, action_dim, writer, epsilon_decay=0.01)
    madqn2 = MADQN(n_agents, state_dim, action_dim, writer, epsilon=0, gamma=0)
    
    best_avg_reward = float('-inf')
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_rewards = {i: 0 for i in range(n_cars)}
        episode_steps = 0
        
        for step in range(max_steps):
            episode_steps += 1
            states = {i: obs[i].flatten() for i in range(n_cars)}
            actions = madqn.select_action(states)
            actions2 = madqn2.select_action(states)

            actions[2] = actions2[0]
            actions[3] = actions2[1]

            next_obs, rewards, dones, info = env.step(actions)
            
            for agent_id in range(n_agents):
                madqn.replay_buffers[agent_id].push(
                    states[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_obs[agent_id].flatten(),
                    dones[agent_id]
                )
                episode_rewards[agent_id] = rewards[agent_id]

            for enemy_id in range(n_agents):
                i = enemy_id + 2
                madqn2.replay_buffers[enemy_id].push(
                    states[i],
                    actions[i],
                    rewards[i],
                    next_obs[i].flatten(),
                    dones[i]
                )
                episode_rewards[i] = rewards[i]
            
            for agent_id in range(n_agents):
                madqn.update(agent_id)
            
            for enemy_id in range(n_agents):
                madqn2.update(enemy_id)

            obs = next_obs
            
            if any(dones.values()):
                break
        
        # Log metrics

        #Log episode metrics to TensorBoard
        for agent_id in range(n_cars):
            writer.add_scalar(f'Rewards/agent_{agent_id}', episode_rewards[agent_id], episode)
            writer.add_scalar(f'Steps/agent_{agent_id}', episode_steps, episode)
        
        writer.add_scalar('Training/epsilon', madqn.epsilon, episode)
        writer.add_scalar('Training/episode_length', episode_steps, episode)

        avg_reward = sum(episode_rewards.values()) / n_agents
        writer.add_scalar('Rewards/average', avg_reward, episode)

        # Assuming episode_rewards is a dictionary with agent_id as keys and reward as values
        first_two_agents = list(episode_rewards.values())[:2]

        # Compute the average reward for the first two agents
        avg_reward_first_two_agents = sum(first_two_agents) / len(first_two_agents)

        # Log checkpoints reached and other custom metrics if available in info
        if info:
            for agent_id in range(n_agents):
                if 'checkpoints_reached' in info:
                    writer.add_scalar(f'Checkpoints/agent_{agent_id}', 
                                    info['checkpoints_reached'].get(agent_id, 0), 
                                    episode)
        
        # Save checkpoints
        if (episode + 1) % save_frequency == 0:
            madqn.save_model(save_dir, episode + 1)
        
        # Save best model
        if avg_reward_first_two_agents > best_avg_reward:
            best_avg_reward = avg_reward_first_two_agents
            madqn.save_model(os.path.join(save_dir, 'best_model'), episode + 1)
        
        print(f"Episode {episode + 1}")
        for agent_id in range(madqn.n_agents):
            print(f"Agent {agent_id} total reward: {episode_rewards[agent_id]}")
        print(f"Average Reward: {avg_reward_first_two_agents:.2f}")
        print(f"Epsilon: {madqn.epsilon}")
        print("--------------------")
    
    writer.close()
    return madqn

def test_madqn(checkpoint_path, n_episodes=100, render=True):
    """Test a trained MADQN model"""
    # Create unique run name with timestamp
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'MADQN_{current_time}')
    writer = SummaryWriter(log_dir)
    # Load configuration
    config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize environment and agent
    env = MultiCarRacing(
        n_cars=4,
        grid_size=30,
        track_width=5,
        render_mode='human' if render else None
    )
    
    madqn = MADQN(
        writer=writer,
        n_agents=config['n_agents'],
        state_dim=config['state_dim'],
        action_dim=config['action_dim']
    )
    
    # Load trained model
    madqn.load_model(checkpoint_path)
    
    # Set epsilon to minimum for testing (mostly exploiting)
    madqn.epsilon = 0.01
    
    total_rewards = []
    checkpoints_reached = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_rewards = {i: 0 for i in range(madqn.n_agents)}
        done = False
        
        while not done:
            states = {i: obs[i].flatten() for i in range(madqn.n_agents)}
            actions = madqn.select_action(states)
            
            action_map = {
                pygame.K_LEFT: 0,  # Left arrow
                pygame.K_RIGHT: 1,  # Right arrow
                pygame.K_UP: 3,  # Up arrow
                pygame.K_DOWN: 2,  # Down arrow
            }
            actions[2] = 4
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                 if event.key in action_map:
                     actions[2] = action_map[event.key]
            
            actions[3] = np.random.randint(0, 5)
            obs, rewards, dones, info = env.step(actions)

            print(f'angle: {env.agents[2].angle}')

            for agent_id in range(madqn.n_agents):
                episode_rewards[agent_id] = rewards[agent_id]
            
            if render:
                env.render()
            
            done = any(dones.values())
        
        avg_reward = sum(episode_rewards.values()) / madqn.n_agents
        total_rewards.append(avg_reward)
        
        if 'checkpoints_reached' in info:
            checkpoints_reached.append(sum(info['checkpoints_reached'].values()) / madqn.n_agents)
        
        print(f"Episode {episode + 1}")
        for agent_id in range(madqn.n_agents):
            print(f"Agent {agent_id} total reward: {episode_rewards[agent_id]}")
        # print(f"Average Reward: {avg_reward:.2f}")
        if checkpoints_reached:
            print(f"Average Checkpoints: {checkpoints_reached[-1]:.2f}")
        print("--------------------")
    
    # Print test results
    print("\nTest Results:")
    print(f"Average Reward over {n_episodes} episodes: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    if checkpoints_reached:
        print(f"Average Checkpoints: {np.mean(checkpoints_reached):.2f} ± {np.std(checkpoints_reached):.2f}")


if __name__ == "__main__":
    # train_madqn()
    # Training
    madqn = train_madqn(n_episodes=1000, max_steps=2000, save_dir='checkpoints', save_frequency=100)
    
    # Testing
    # Uncomment and modify path to test a specific checkpoint
    # test_madqn('checkpoints/best_model/checkpoint_episode_21.pth', n_episodes=10, render=True)
    