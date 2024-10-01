from env import ModTSP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AgentQLearner:
    def __init__(self, action_count, lr=0.001, discount=0.99):
        """Initialize the Q-learning agent."""
        self.action_count = action_count  # Number of actions (targets)
        self.lr = lr  # Learning rate
        self.discount = discount  # Discount factor for future rewards
        self.q_values = {}  # Q-value table to store state-action values
        self.exploration_rate = 1.0  # Initial exploration rate
        self.min_exploration_rate = 0.01  # Minimum exploration rate for epsilon-greedy
        self.exploration_decay = 0.995  # Decay rate for exploration

    def choose_action(self, current_state):
        """Select an action using epsilon-greedy policy."""
        state_key = str(current_state)
        # Initialize Q-values for unseen state
        if state_key not in self.q_values:
            self.q_values[state_key] = np.zeros(self.action_count)
         
        visited_state = current_state[1:11]
        
        allowed = [i for i,s in enumerate(visited_state) if s == 0]   
        
        
        # Exploration vs. Exploitation decision
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(allowed) if allowed else np.random.choice(self.action_count)  # Explore
        else: 
            qvals = self.q_values[state_key].copy()
            for id, visited in enumerate(visited_state):
                if visited == 1:
                    qvals[id] = float("-inf")
        return np.argmax(qvals)  # Exploit

    def update_q_table(self, current_state, selected_action, reward_obtained, next_state):
        """Apply Q-learning update rule to adjust Q-values."""
        state_key = str(current_state)
        next_state_key = str(next_state)

        # Initialize Q-values for unseen next state
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = np.zeros(self.action_count)

        # Q-learning update: Q(s, a) = Q(s, a) + lr * (reward + discount * max Q(s', a') - Q(s, a))
        best_future_action = np.argmax(self.q_values[next_state_key])
        q_target = reward_obtained + self.discount * self.q_values[next_state_key][best_future_action]
        q_delta = q_target - self.q_values[state_key][selected_action]
        self.q_values[state_key][selected_action] += self.lr * q_delta

        # Decay exploration rate to reduce exploration over time
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)


def render_path(environment, learner, path_traversed, save_file="D:/tsp_path_render.mp4"):
    """Animate and visualize the agent's path through the environment."""
    fig, ax = plt.subplots()
    ax.scatter(environment.locations[:, 0], environment.locations[:, 1], c='red', label='Waypoints')  # Target locations
    trajectory_line, = ax.plot([], [], 'b-', lw=2, label='Agent Path')  # Line for the agent's path

    def initialize():
        trajectory_line.set_data([], [])
        return trajectory_line,

    def animate(frame_idx):
        """Update the agent's path step by step."""
        visited_coords = np.array([environment.locations[pt] for pt in path_traversed[:frame_idx+1]])
        trajectory_line.set_data(visited_coords[:, 0], visited_coords[:, 1])
        return trajectory_line,

    # Create animation
    path_animation = animation.FuncAnimation(fig, animate, frames=len(path_traversed), init_func=initialize, blit=True, repeat=False)
    plt.legend()
    # Save the animation to the specified file
    path_animation.save(save_file, writer='ffmpeg', fps=2)
    plt.show()


def run_simulation():
    """Run the Q-learning agent in the custom TSP environment."""
    num_points = 10
    reshuffle_time = 10
    total_episodes = 10000  # Number of episodes to run the Q-learning agent

    # Initialize environment and agent
    environment = ModTSP(num_points, shuffle_time=reshuffle_time)
    learner = AgentQLearner(action_count=num_points)

    rewards_collected = []
    exploration_rates = []
    episode_paths = [] 
    episode_results = {}

    for ep in range(total_episodes):
        state, _ = environment.reset()
        done_flag = False
        cumulative_reward = 0
        visited_path = []

        while not done_flag:
            action = learner.choose_action(state)
            visited_path.append(action)  
            next_state, reward, end_condition, time_up, _ = environment.step(action)
            done_flag = end_condition or time_up
            learner.update_q_table(state, action, reward, next_state)

            cumulative_reward += reward
            state = next_state
        episode_paths.append(visited_path)
        rewards_collected.append(cumulative_reward)
        exploration_rates.append(np.mean(learner.exploration_rate))

        if ep % 100 == 0:
            episode_results[f"Rewards after episode {ep}"] = cumulative_reward

    print(episode_results)
    plt.plot(rewards_collected, color='blue', label='Total Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Episode Progression vs Cumulative Reward')
    plt.legend()
    plt.show()
    render_path(environment, learner, episode_paths[-1])


if __name__ == "__main__":
    run_simulation()
