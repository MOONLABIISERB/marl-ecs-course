import matplotlib.pyplot as plt
import numpy as np



class MAPFEnv:
    def __init__(self, 
                grid_size, 
                seed,
                mode) -> None:
        '''
        Initialize the MAPF environment

        Parameters:
        grid_size: int, size of the grid
        seed: int, random seed for reproducibility
        mode: str, mode for agent initialization

        Returns:
        None
        '''
        self.grid_size = grid_size
        self.walls = [(5,0), (5,1), (5,2), (4,2),
                      (0,5), (1,5), (2,5), (2,4),
                      (4,9), (4,8), (4,7), (5,7),
                      (7,4), (7,5), (8,5), (9,5)]   

        self.agent_colors = ['blue', 'yellow', 'violet', 'green']
        self.goal_pos = [(5,8), (1,4), (4,1), (8,4)]
        self.current_pos = None
        self.mode = mode
        
        if seed is not None:
            np.random.seed(seed)


    def reset(self) -> list:
        '''
        Reset the environment and return the initial positions of the agents

        Parameters:
        None

        Returns:
        list: list of initial positions of the agents
        '''
        if self.mode == 'random':
            self.current_pos = [(np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)) for _ in range(4)]
        else:
            self.current_pos = [(1,1), (8,1), (8,8), (1,8)]
        return self.current_pos


    def step(self, actions: list) -> tuple:
        '''
        Take a step in the environment

        Parameters:
        actions: list, actions taken by each agent

        Returns:
        tuple: next_positions, rewards, done, {}
        '''
        rewards = []
        next_positions = []
    
        for i, (pos, action) in enumerate(zip(self.current_pos, actions)):
            next_pos = self._get_next_position(pos, action)

            if self._is_valid_move(next_pos, next_positions):  
                next_positions.append(next_pos)
            else:
                next_positions.append(pos)

            reward = 0 if next_pos == self.goal_pos[i] else -1
            rewards.append(reward)

        self.current_pos = next_positions
        done = all([pos == goal for pos, goal in zip(next_positions, self.goal_pos)])
        return self.current_pos, rewards, done, {}



    def _get_next_position(self, pos: tuple, action: int) -> tuple:
        '''
        Get the next position based on the current position and action

        Parameters:
        pos: tuple, current position
        action: int, action taken by the agent
        
        Returns:
        tuple: next position
        '''
        x, y = pos
        if action == 0:   # Left
            return (x, max(0, y-1))
        elif action == 1: # Right
            return (x, min(self.grid_size-1, y+1))
        elif action == 2: # Up
            return (max(0, x-1), y)
        elif action == 3: # Down
            return (min(self.grid_size-1, x+1), y)
        return (x, y) #Stay


    def _is_valid_move(self, pos: tuple, current_next_positions: list) -> bool:
        '''
        Check if the move is valid

        Parameters:
        pos: tuple, position to check
        current_next_positions: list, list of next positions of all agents

        Returns:
        bool: True if move is valid, False otherwise
        '''
        if pos in self.walls:
            return False

        if pos in current_next_positions:
            return False

        return True
    

    def plot_map(self):
        agents_pos = [(1,1), (8,1), (1,8), (8,8)]
        fig, ax = plt.subplots(figsize=(8, 8))
        # Set up gridlines and limits
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        ax.grid(True)
        # Set aspect of the plot to be equal
        ax.set_aspect('equal')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        # Remove the axes
        ax.set_xticklabels(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticklabels(np.arange(0, self.grid_size + 1, 1))
        ax.tick_params(left=False, bottom=False)
        for (x,y) in self.walls:
            ax.add_patch(plt.Rectangle((x,y), 1, 1, color='grey'))
        for i, (x,y) in enumerate(agents_pos):
            color = self.agent_colors[i]
            ax.add_patch(plt.Rectangle((x,y), 1, 1, color=color))
            ax.text(x+0.5, y+0.5, str(i), color='black', ha='center', va='center')
        
        for i, (x,y) in enumerate(self.goal_pos):
            color = self.agent_colors[i]
            plt.plot(x+0.5, y+0.5, marker='+', color=color, mew=2, ms=10)
        plt.savefig('mapf_env_fixed.png')





class QLearningAgent:
    def __init__(self, 
                state_size: int, 
                n_actions: int, 
                learning_rate=0.03, 
                discount_factor=0.99, 
                epsilon=0.1) -> None:
        '''
        Initialize the QLearning agent

        Parameters:
        state_size: int, size of the state space
        n_actions: int, number of actions
        learning_rate: float, learning rate for Q-learning
        discount_factor: float, discount factor for Q-learning
        epsilon: float, epsilon-greedy parameter

        Returns:
        None
        '''
        self.q_table = np.zeros((state_size, state_size, n_actions))
        self.policy = np.zeros((state_size, state_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.state_size = state_size


    def get_action(self, state: tuple) -> int:
        '''
        Get the action to take based on the current state

        Parameters:
        state: tuple, current state

        Returns:
        int: action to take
        '''
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)  # Random action
        return self.policy[state[0], state[1]]


    def update(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        '''
        Update the Q-values based on the current state, action, reward, and next state

        Parameters:
        state: tuple, current state
        action: int, action taken
        reward: float, reward received
        next_state: tuple, next state

        Returns:
        None
        '''
        action = int(action)
        curr_x, curr_y = map(int, state)
        next_x, next_y = map(int, next_state)
        
        current_q = self.q_table[curr_x, curr_y, action]
        next_max_q = np.max(self.q_table[next_x, next_y])
        td_error = reward + self.gamma * next_max_q - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[curr_x, curr_y, action] = new_q
        self.policy[curr_x, curr_y] = np.argmax(self.q_table[curr_x, curr_y])



def train(seed: int) -> tuple:
    '''
    Train the Q-learning agents

    Parameters:
    seed: int, random seed for reproducibility

    Returns:
    tuple: agent_rewards, agents, env
    '''
    n_episodes = 8000
    max_steps = 3000
    grid_size = 10
    n_agents = 4
    
    mode = input("Enter mode (random or None): ")

    env = MAPFEnv(grid_size, seed, mode)
    agents = [QLearningAgent(grid_size, 5) for _ in range(n_agents)]
    
    agent_rewards = np.zeros((n_episodes, n_agents))
    cumulative_rewards = np.zeros((n_episodes, n_agents))
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = np.zeros(n_agents)
        
        for step in range(max_steps):
            actions = [agent.get_action(pos) for agent, pos in zip(agents, state)]
            next_state, rewards, done, _ = env.step(actions)
            
            # Update each agent's Q-values
            for i in range(n_agents):
                agents[i].update(
                    state[i], 
                    actions[i], 
                    rewards[i], 
                    next_state[i]
                )
            
            episode_reward += rewards
            state = next_state
            
            if done:
                break
        
        agent_rewards[episode] = episode_reward
        
        # Update cumulative rewards
        if episode == 0:
            cumulative_rewards[episode] = episode_reward
        else:
            cumulative_rewards[episode] = cumulative_rewards[episode-1] + episode_reward
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_reward):.2f}")
    
    # Plot both episode rewards and cumulative rewards
    plt.figure(figsize=(12, 6))
    for i in range(n_agents):
        plt.plot(agent_rewards[:, i], alpha=0.4, label=f'Agent {i} Episode Reward')
        plt.plot(cumulative_rewards[:, i]/(np.arange(n_episodes)+1), 
                label=f'Agent {i} Cumulative Avg Reward')
    
    plt.title('Agent Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('agent_rewards_random.png')
    plt.show()
    
    return agent_rewards, agents, env


def find_min_steps_for_all_agents(env: MAPFEnv, agents: list) -> np.ndarray:
    '''
    Find the minimum number of steps for each agent to reach its goal

    Parameters:
    env: MAPFEnv, environment object
    agents: list, list of agents

    Returns:
    np.ndarray: array of steps for each agent
    '''
    steps_to_goal = np.zeros(len(agents))  # Store steps for each agent
    
    for i, agent in enumerate(agents):
        state = env.reset()  # Reset environment and get initial positions of all agents
        agent_position = state[i]  # Get the starting position of this agent
        steps = 0
        
        while agent_position != env.goal_pos[i]:  # Loop until the agent reaches its goal
            action = agent.get_action(agent_position)
            next_state, _, _, _ = env.step([action if j == i else 0 for j in range(len(agents))])  # Get next state
            agent_position = next_state[i]  # Update agent's position
            steps += 1
        
        steps_to_goal[i] = steps  # Store the number of steps for this agent
    
    return steps_to_goal



SEED = 42
agent_rewards, agents, env = train(seed=SEED)

steps = find_min_steps_for_all_agents(env, agents)
print(f"Minimum steps for each agent to reach their goal: {steps}")



def save_q_values(agents, filename='q_values.npz') -> None:
    """
    Save Q-values for all agents to a file
    """
    q_values = {f'agent_{i}': agent.q_table for i, agent in enumerate(agents)}
    np.savez(filename, **q_values)

save_q_values(agents, 'q_values.npz')

env.plot_map()

