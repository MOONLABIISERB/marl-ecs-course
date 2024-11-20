import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class Environment:
    def __init__(self, grid_size: int, seed: int) -> None:
        """
        Initialize the Multi-Agent Pathfinding environment

        Parameters:
        grid_size: int, size of the grid
        seed: int, random seed for reproducibility
        """
        self.grid_size = grid_size
        np.random.seed(seed)
        
        self.walls = [
            (5,0), (5,1), (5,2), (4,2),
            (0,5), (1,5), (2,5), (2,4),
            (4,9), (4,8), (4,7), (5,7),
            (7,4), (7,5), (8,5), (9,5)
        ]   

        self.agent_colors = ['coral', 'skyblue', 'mediumorchid', 'lightgreen']
        self.goal_positions = [(5,8), (1,4), (4,1), (8,4)]
        self.current_positions = None


    def reset(self) -> list:
        """
        Reset the environment and return initial agent positions

        Returns:
        list: list of initial positions of the agents
        """
        self.current_positions = [(1,1), (8,1), (8,8), (1,8)]
        return self.current_positions


    def step(self, actions: list) -> tuple:
        """
        Take a step in the environment

        Parameters:
        actions: list, actions taken by each agent

        Returns:
        tuple: next_positions, rewards, done, {}
        """
        next_positions = []
        rewards = []
    
        for i, (pos, action) in enumerate(zip(self.current_positions, actions)):
            next_pos = self._compute_next_position(pos, action)

            if self._is_valid_move(next_pos, next_positions):  
                next_positions.append(next_pos)
            else:
                next_positions.append(pos)

            reward = 0 if next_pos == self.goal_positions[i] else -1
            rewards.append(reward)

        self.current_positions = next_positions
        done = all([pos == goal for pos, goal in zip(next_positions, self.goal_positions)])
        
        return self.current_positions, rewards, done, {}


    def _compute_next_position(self, pos: tuple, action: int) -> tuple:
        """
        Compute next position based on current position and action

        Parameters:
        pos: tuple, current position
        action: int, action taken by the agent
        
        Returns:
        tuple: next position
        """
        x, y = pos
        if action == 0:   # Left
            return (x, max(0, y-1))
        elif action == 1: # Right
            return (x, min(self.grid_size-1, y+1))
        elif action == 2: # Up
            return (max(0, x-1), y)
        elif action == 3: # Down
            return (min(self.grid_size-1, x+1), y)
        return (x, y) # Stay


    def _is_valid_move(self, pos: tuple, current_next_positions: list) -> bool:
        """
        Check if the move is valid

        Parameters:
        pos: tuple, position to check
        current_next_positions: list, list of next positions of all agents

        Returns:
        bool: True if move is valid, False otherwise
        """
        return pos not in self.walls and pos not in current_next_positions


class QLearningAgent:
    def __init__(self, state_size: int, n_actions: int, 
                 learning_rate=0.03, 
                 discount_factor=0.99, 
                 epsilon=0.1) -> None:
        """
        Initialize the Q-Learning agent

        Parameters:
        state_size: int, size of the state space
        n_actions: int, number of actions
        learning_rate: float, learning rate
        discount_factor: float, discount factor
        epsilon: float, exploration rate
        """
        self.q_table = np.zeros((state_size, state_size, n_actions))
        self.policy = np.zeros((state_size, state_size), dtype=int)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.state_size = state_size


    def get_action(self, state: tuple) -> int:
        """
        Select action based on epsilon-greedy policy

        Parameters:
        state: tuple, current state

        Returns:
        int: action to take
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)
        return self.policy[state[0], state[1]]


    def update(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        """
        Update Q-values using Q-learning algorithm

        Parameters:
        state: tuple, current state
        action: int, action taken
        reward: float, reward received
        next_state: tuple, next state
        """
        curr_x, curr_y = map(int, state)
        next_x, next_y = map(int, next_state)
        
        current_q = self.q_table[curr_x, curr_y, action]
        next_max_q = np.max(self.q_table[next_x, next_y])
        td_error = reward + self.gamma * next_max_q - current_q
        new_q = current_q + self.lr * td_error
        
        self.q_table[curr_x, curr_y, action] = new_q
        self.policy[curr_x, curr_y] = np.argmax(self.q_table[curr_x, curr_y])


    def set_q_table(self, q_table):
        """
        Set the Q-table for the agent

        Parameters:
        q_table: np.ndarray, the Q-table to load
        """
        self.q_table = q_table
        self.policy = np.argmax(self.q_table, axis=2)


def train(seed: int = 42, n_episodes: int = 20000, max_steps: int = 10000) -> tuple:
    """
    Train Q-learning agents for Multi-Agent Pathfinding

    Parameters:
    seed: int, random seed
    n_episodes: int, number of training episodes
    max_steps: int, maximum steps per episode

    Returns:
    tuple: agent_rewards, agents, environment
    """
    grid_size = 10
    n_agents = 4

    env = Environment(grid_size, seed)
    agents = [QLearningAgent(grid_size, 5) for _ in range(n_agents)]
    
    agent_rewards = np.zeros((n_episodes, n_agents))
    cumulative_rewards = np.zeros((n_episodes, n_agents))
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = np.zeros(n_agents)
        
        for _ in range(max_steps):
            actions = [agent.get_action(pos) for agent, pos in zip(agents, state)]
            next_state, rewards, done, _ = env.step(actions)
            
            for i in range(n_agents):
                agents[i].update(state[i], actions[i], rewards[i], next_state[i])
            
            episode_reward += rewards
            state = next_state
            
            if done:
                break
        
        agent_rewards[episode] = episode_reward
        
        cumulative_rewards[episode] = (cumulative_rewards[episode-1] + episode_reward) if episode > 0 else episode_reward
        
        if episode % 500 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_reward):.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn')
    
    for i in range(n_agents):
        plt.plot(agent_rewards[:, i], alpha=0.4, label=f'Agent {i} Episode Reward')
        plt.plot(cumulative_rewards[:, i]/(np.arange(n_episodes)+1), 
                label=f'Agent {i} Cumulative Avg Reward')
    
    plt.title('Learning Performance of Each Agent', fontsize=15)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('all_agents_performance.png')
    plt.show()
    
    return agent_rewards, agents, env


def compute_min_steps_to_goal(env, agents):
    """
    Compute minimum steps for each agent to reach its goal

    Parameters:
    env: Environment object
    agents: list of QLearningAgent objects

    Returns:
    np.ndarray: Minimum steps for each agent
    """
    steps_to_goal = np.zeros(len(agents))  
    
    for i, agent in enumerate(agents):
        state = env.reset()
        agent_position = state[i]
        steps = 0
        
        while agent_position != env.goal_positions[i]:
            action = agent.get_action(agent_position)
            next_state, _, _, _ = env.step([action if j == i else 0 for j in range(len(agents))])
            agent_position = next_state[i]
            steps += 1
        
        steps_to_goal[i] = steps
    
    return steps_to_goal


def save_q_values(agents, filename='q_values.npz'):
    """
    Save Q-values for all agents to a file

    Parameters:
    agents: list of agents
    filename: str, output filename
    """
    q_values = {f'agent_{i}': agent.q_table for i, agent in enumerate(agents)}
    np.savez(filename, **q_values)


def load_q_values(filename='q_values.npz', grid_size=10, n_actions=5):
    """
    Load Q-values for all agents from a file

    Parameters:
    filename: str, input filename
    grid_size: int, size of grid
    n_actions: int, number of possible actions

    Returns:
    list: Loaded agents
    """
    data = np.load(filename)
    agents = []
    
    for i in range(len(data.files)):
        agent = QLearningAgent(state_size=grid_size, n_actions=n_actions)
        agent.set_q_table(data[f'agent_{i}'])
        agents.append(agent)
    
    return agents


def create_optimal_path_gif(env, agents, filename="optimal_paths.gif", max_steps=20):
    """
    Create a visualization of optimal paths as a GIF

    Parameters:
    env: Environment object
    agents: list of trained agents
    filename: str, output GIF filename
    max_steps: int, maximum simulation steps
    """
    state = env.reset()
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.style.use('seaborn')
    
    def init():
        ax.clear()
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_xticks(np.arange(0, env.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, env.grid_size + 1, 1))
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
        return []

    def update(frame):
        nonlocal state
        
        ax.clear()
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_xticks(np.arange(0, env.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, env.grid_size + 1, 1))
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')

        # Draw walls
        for x, y in env.walls:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='lightgrey'))

        # Draw goals
        for i, (x, y) in enumerate(env.goal_positions):
            ax.plot(x + 0.5, y + 0.5, marker='+', color=env.agent_colors[i], mew=2, ms=10)
        
        # Draw agents
        for i, (x, y) in enumerate(state):
            color = env.agent_colors[i]
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, alpha=0.7))
            ax.text(x + 0.5, y + 0.5, str(i), color='black', ha='center', va='center')

        actions = [agent.get_action(pos) for agent, pos in zip(agents, state)]
        next_state, _, done, _ = env.step(actions)
        state = next_state

        if done:
            ani.event_source.stop()

        return []

    ani = FuncAnimation(fig, update, frames=max_steps, init_func=init, blit=True, interval=500)
    ani.save(filename, writer=PillowWriter(fps=2))
    print(f"GIF saved as {filename}")


# Main execution
if __name__ == "__main__":
    SEED = 42
    agent_rewards, agents, env = train(seed=SEED)

    steps = compute_min_steps_to_goal(env, agents)
    print(f"Minimum steps for each agent to reach their goal: {steps}")

    save_q_values(agents, 'q_table.npz')
    loaded_agents = load_q_values(filename='q_table.npz', grid_size=10)
    create_optimal_path_gif(env, loaded_agents, filename='optimal_paths.gif')
