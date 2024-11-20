import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

class MAPFEnv:
    def __init__(self, grid_size, seed, mode) -> None:
        '''
        Initialize the Multi-Agent Pathfinding (MAPF) environment.

        Parameters:
        grid_size: int, size of the grid (NxN)
        seed: int, random seed for reproducibility
        mode: str, initialization mode for agents (1 for fixed, 2 for random)

        Returns:
        None
        '''
        self.grid_size = grid_size
        self.walls = [(5,0), (5,1), (5,2), (4,2),
                      (0,5), (1,5), (2,5), (2,4),
                      (4,9), (4,8), (4,7), (5,7),
                      (7,4), (7,5), (8,5), (9,5)]   

        self.agent_colors = ['red', 'yellow', 'blue', 'black']
        self.goal_pos = [(5,8), (1,4), (4,1), (8,4)]
        self.current_pos = None
        self.mode = mode

        if seed is not None:
            np.random.seed(seed)

    def reset(self) -> list:
        '''
        Reset the environment to its initial state.

        Parameters:
        None

        Returns:
        list: Initial positions of the agents.
        '''
        if self.mode == 2:
            self.current_pos = [(np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)) for _ in range(4)]
        else:  # Mode 1
            self.current_pos = [(1, 1), (8, 1), (8, 8), (1, 8)]
        return self.current_pos

    def step(self, actions: list) -> tuple:
        '''
        Execute a step in the environment.

        Parameters:
        actions: list, actions selected by each agent.

        Returns:
        tuple: Updated positions, rewards, done flag, and additional info (empty dictionary).
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
        Compute the next position based on the current position and action.

        Parameters:
        pos: tuple, current position
        action: int, action selected by the agent
        
        Returns:
        tuple: Next position after taking the action.
        '''
        x, y = pos
        if action == 0:   # Move left
            return (x, max(0, y-1))
        elif action == 1:  # Move right
            return (x, min(self.grid_size-1, y+1))
        elif action == 2:  # Move up
            return (max(0, x-1), y)
        elif action == 3:  # Move down
            return (min(self.grid_size-1, x+1), y)
        return (x, y)  # Stay in the current position

    def _is_valid_move(self, pos: tuple, current_next_positions: list) -> bool:
        '''
        Verify if the selected move is valid.

        Parameters:
        pos: tuple, proposed new position
        current_next_positions: list, positions already planned for other agents.

        Returns:
        bool: True if the move is valid, False otherwise.
        '''
        if pos in self.walls or pos in current_next_positions:
            return False
        return True


class QLearningAgent:
    def __init__(self, state_size: int, n_actions: int, learning_rate=0.03, discount_factor=0.99, epsilon=0.1) -> None:
        '''
        Initialize a Q-learning agent.

        Parameters:
        state_size: int, dimensions of the state space
        n_actions: int, number of possible actions
        learning_rate: float, step size for Q-value updates
        discount_factor: float, future reward discount factor
        epsilon: float, exploration-exploitation tradeoff parameter

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
        Select an action based on the agent's policy or exploration.

        Parameters:
        state: tuple, current state of the agent

        Returns:
        int: Selected action.
        '''
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)  # Random action
        return self.policy[state[0], state[1]]

    def update(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        '''
        Update the Q-values and policy using the Q-learning update rule.

        Parameters:
        state: tuple, current state
        action: int, action taken
        reward: float, reward received for the action
        next_state: tuple, resulting state after the action

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

    def set_q_table(self, q_table):
        '''
        Load a pre-trained Q-table for the agent.

        Parameters:
        q_table: np.ndarray, Q-table to be loaded.

        Returns:
        None
        '''
        self.q_table = q_table
        self.policy = np.argmax(self.q_table, axis=2)


def train(seed: int) -> tuple:
    '''
    Train Q-learning agents in the MAPF environment.

    Parameters:
    seed: int, random seed for reproducibility.

    Returns:
    tuple: Episode rewards, trained agents, and environment instance.
    '''
    n_episodes = 50000
    max_steps = 10000
    grid_size = 10
    n_agents = 4

    global mode
    mode_input = input("Enter mode (1 for fixed, 2 for random): ")
    mode = int(mode_input)

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
            
            # Update Q-values for all agents
            for i in range(n_agents):
                agents[i].update(state[i], actions[i], rewards[i], next_state[i])
            
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
    
    # Plot rewards over episodes
    plt.figure(figsize=(12, 6))
    for i in range(n_agents):
        plt.plot(agent_rewards[:, i], alpha=0.4, label=f'Agent {i} Episode Reward')
        plt.plot(cumulative_rewards[:, i] / (np.arange(n_episodes) + 1), label=f'Agent {i} Cumulative Avg Reward')
    
    plt.title('Agent Rewards Over Time')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(f'agent_rewards_{mode}.png')
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
        state = env.reset()
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



def create_gif(env: MAPFEnv, agents: list, filename="optimal_paths.gif", max_steps=20):
    """
    Create a GIF of the optimal path for all agents

    Parameters:
    env: MAPFEnv, the MAPF environment
    agents: list, list of QLearningAgent objects with pre-trained Q-tables
    filename: str, name of the output GIF file
    max_steps: int, maximum number of steps to simulate
    """
    state = env.reset()
    frames = []  # To store frames for the animation
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def init():
        """Initialize the plot."""
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_xticks(np.arange(0, env.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, env.grid_size + 1, 1))
        ax.grid(True)
        ax.set_aspect('equal')
        return []
    
    def update(frame):
        """Update the plot for the current frame."""
        nonlocal state

    # Clear the previous frame
        ax.clear()
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_xticks(np.arange(0, env.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, env.grid_size + 1, 1))
        ax.grid(True)
        ax.set_aspect('equal')

    # Draw walls
        for x, y in env.walls:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='grey'))

    # Draw goals
        for i, (x, y) in enumerate(env.goal_pos):
            ax.plot(x + 0.5, y + 0.5, marker='+', color=env.agent_colors[i], mew=2, ms=10)

    # Draw agents with custom labels
        agent_labels = ['A1', 'A2', 'A3', 'A4']  # Custom labels for the agents
        for i, (x, y) in enumerate(state):
            color = env.agent_colors[i]
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))
            ax.text(x + 0.5, y + 0.5, agent_labels[i], color='white', ha='center', va='center')

    # Perform a step
        actions = [agent.get_action(pos) for agent, pos in zip(agents, state)]
        next_state, _, done, _ = env.step(actions)
        state = next_state

    # Stop animation if done
        if done:
            ani.event_source.stop()

        return []

    ani = FuncAnimation(fig, update, frames=max_steps, init_func=init, blit=True, interval=500)
    
    # Save the animation as a GIF
    ani.save(filename, writer=PillowWriter(fps=2))
    print(f"GIF saved as {filename}")



def load_q_values(filename='q_values.npz', grid_size=10, n_actions=5) -> list:
    """
    Load Q-values for all agents from a file
    """
    data = np.load(filename)
    agents = []
    
    for i in range(len(data.files)):
        agent = QLearningAgent(state_size=grid_size, n_actions=n_actions)
        agent.set_q_table(data[f'agent_{i}'])
        agents.append(agent)
    
    return agents


env_test = MAPFEnv(10, SEED, mode)
agents = load_q_values(filename='q_values.npz', grid_size=10)
create_gif(env, agents, filename=f'path_{mode}.gif')