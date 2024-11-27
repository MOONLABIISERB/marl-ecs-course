import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from environment import PressurePlate

# Hyperparameters
n_episodes = 300         
max_steps = 10000        
gridsize = (15, 9)      
n_agents = 4            
n_actions = 5           
a_range = 2             
update_target_frequency = 100  # Frequency of target network updates

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.0, epsilon_min=0.00, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)
        
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Track learning steps for target network updates
        self.learn_step_counter = 0
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def act(self, state):
        # Implement proper exploration strategy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        
        # Exploit: choose best action
        state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        if isinstance(state, tuple):
            state = np.array(state)
        if isinstance(next_state, tuple):
            next_state = np.array(next_state)
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        # Learn more frequently
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Compute current Q values
        q_values = self.model(states).gather(1, actions).squeeze(1)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(q_values, targets)
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network more dynamically
        self.learn_step_counter += 1
        if self.learn_step_counter % update_target_frequency == 0:
            self.update_target_model()

# Training loop
if __name__ == "__main__":
    env = PressurePlate(gridsize[0], gridsize[1], n_agents, a_range, 'linear')
    obs = env.reset()
    
    # Calculate correct state dimension from environment
    state_dim = len(obs[0])  # Get dimension from first agent's observation
    action_dim = 5
    
    # Create agents
    agents = [DQNAgent(state_dim, action_dim) for _ in range(n_agents)]
    
    # Store episodic rewards for each agent
    episodic_rewards = {i: [] for i in range(n_agents)}
    cumulative_rewards = {i: 0 for i in range(n_agents)}
    
    for episode in range(n_episodes):
        states = env.reset()
        total_rewards = [0] * n_agents
        print(f"Episode {episode}-----------------------")
        
        for step in range(max_steps):
            actions = []
            # Get actions for all agents
            for i in range(n_agents):
                action = agents[i].act(states[i])
                actions.append(action)
            
            # Take step in environment
            next_states, rewards, dones, _ = env.step(actions)
            
            # Store transitions for each agent and learn after each step
            for i in range(n_agents):
                agents[i].store_transition(
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i],
                    dones[i]
                )
                # Learn after storing transition
                agents[i].replay()
                total_rewards[i] += rewards[i]
            
            states = next_states
            env.render()
            if all(dones):
                break
        
        # Record episodic rewards
        for i in range(n_agents):
            episodic_rewards[i].append(total_rewards[i])
            cumulative_rewards[i] += total_rewards[i]
        
        # Print episode results
        print(f"Episode {episode+1}/{n_episodes}")
        for i in range(n_agents):
            print(f"Agent {i} Total Reward: {total_rewards[i]:.2f}, Epsilon: {agents[i].epsilon:.4f}")
    
    # Plotting Results
    plt.figure(figsize=(10, 6))
    
    for i in range(n_agents):
        plt.plot(range(n_episodes), episodic_rewards[i], label=f'Agent {i} Episodic Reward', linestyle='-', marker='o')
    
    plt.title("Episodic Rewards vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("combined_reward_plot.png")
    plt.show()