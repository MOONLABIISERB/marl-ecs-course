import numpy as np
from collections import defaultdict
from bonus_env import maze, destinations, agents, maze_size
import matplotlib.pyplot as plt
import pickle 

# Actions: [Stay, Up, Down, Left, Right]
ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ['Stay', 'Up', 'Down', 'Left', 'Right']
NUM_ACTIONS = len(ACTIONS)
GAMMA = 0.9  # Discount factor
ALPHA = 0.1  # Learning rate
EPSILON = 0.2
MAX_EPISODES = 500

# Default Q-Table initializer
def default_q_table():
    return np.zeros(NUM_ACTIONS)

# Maze environment
class MultiAgentMazeMinMax:
    def __init__(self, maze, agents, destinations):
        self.maze = maze
        self.agents = agents
        self.destinations = destinations
        self.num_agents = len(agents)
        self.state = tuple(agents)
        self.steps_taken = [0] * self.num_agents  # Track steps per agent

    def is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < maze_size and 0 <= y < maze_size and self.maze[x, y] != 1

    def reset(self):
        self.state = tuple(agents)
        self.steps_taken = [0] * self.num_agents
        return self.state

    def step(self, actions):
        next_state = list(self.state)
        rewards = []
        for i, action in enumerate(actions):
            current_pos = self.state[i]
            move = ACTIONS[action]
            new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

            # Check if new position is valid
            if self.is_valid_position(new_pos) and new_pos not in next_state:
                next_state[i] = new_pos

            # Increment step count
            self.steps_taken[i] += 1

            # Calculate reward
            if new_pos == self.destinations[i]:
                reward = 10  # Reached goal
            else:
                reward = -1  # Step penalty
            rewards.append(reward)

        self.state = tuple(next_state)
        done = all(next_state[i] == self.destinations[i] for i in range(self.num_agents))
        return self.state, rewards, done, max(self.steps_taken)

# Q-learning with max-time minimization
def train_agents_minmax(maze_env):
    q_tables = [defaultdict(default_q_table) for _ in range(maze_env.num_agents)]
    min_max_time = float('inf')  # Track the minimum maximum time
    rewards_history = []
    steps_history = []

    for episode in range(MAX_EPISODES):
        state = maze_env.reset()
        done = False
        cumulative_rewards = 0
        steps = 0

        while not done:
            actions = []
            for i in range(maze_env.num_agents):
                if np.random.rand() < EPSILON:
                    action = np.random.choice(NUM_ACTIONS)  # Explore
                else:
                    action = np.argmax(q_tables[i][state[i]])  # Exploit
                actions.append(action)

            next_state, rewards, done, max_time = maze_env.step(actions)

            # Update Q-values
            for i in range(maze_env.num_agents):
                current_q = q_tables[i][state[i]][actions[i]]
                next_max_q = np.max(q_tables[i][next_state[i]])
                q_tables[i][state[i]][actions[i]] = current_q + ALPHA * (
                    rewards[i] + GAMMA * next_max_q - current_q
                )

            # Track cumulative rewards and steps
            cumulative_rewards += sum(rewards)
            steps += 1
            state = next_state

        # Update minimum max time
        min_max_time = min(min_max_time, max_time)

        # Record rewards and steps for this episode
        rewards_history.append(cumulative_rewards)
        steps_history.append(steps)

    return q_tables, min_max_time, rewards_history, steps_history

# Save Q-Tables
def save_q_tables(q_tables, filename):
    with open(filename, 'wb') as f:
        pickle.dump(q_tables, f)
    print(f"Q-tables saved to {filename}")

# Plot Rewards and Steps
def plot_training_metrics(rewards_history, steps_history):
    plt.figure(figsize=(12, 5))

    # Plot cumulative rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, label="Cumulative Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.title("Rewards Over Episodes")
    plt.legend()

    # Plot steps
    plt.subplot(1, 2, 2)
    plt.plot(steps_history, label="Steps Per Episode", color='orange')
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.title("Steps Over Episodes")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Train and evaluate
maze_env = MultiAgentMazeMinMax(maze, agents, destinations)
q_tables, min_max_time, rewards_history, steps_history = train_agents_minmax(maze_env)

# Save Q-tables
save_q_tables(q_tables, "q_tables.pkl")

# Visualize rewards and steps
plot_training_metrics(rewards_history, steps_history)

print(f"\nMinimum maximum time achieved: {min_max_time}")
