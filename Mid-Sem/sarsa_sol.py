from tsp_env import ModTSP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate

    def choose_action(self, state):
        """Epsilon-greedy policy for action selection."""
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(self.n_actions)
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)  # Explore
        return np.argmax(self.q_table[state_str])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value using Q-learning update rule."""
        state_str = str(state)
        next_state_str = str(next_state)

        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(self.n_actions)

        best_next_action = np.argmax(self.q_table[next_state_str])
        td_target = reward + self.gamma * self.q_table[next_state_str][best_next_action]
        td_error = td_target - self.q_table[state_str][action]
        self.q_table[state_str][action] += self.alpha * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def animate_solution(env,agent,visited_states,save_path = "D:/tsp_path_Traversal.mp4" ):
    """Animate the agent's traversal through the environment."""
    fig, ax = plt.subplots()
    ax.scatter(env.locations[:, 0], env.locations[:, 1], c='red', label='Targets')
    line, = ax.plot([], [], 'b-', lw=2, label='Path')

    def init():
        line.set_data([], [])
        return line,

    def update(i):
        "updating the agents visited path"
        visited_locs = np.array([env.locations[loc] for loc in visited_states[:i+1]])
        line.set_data(visited_locs[:, 0], visited_locs[:, 1])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(visited_states), init_func=init, blit=True, repeat=False)
    plt.legend()
    ani.save(save_path, writer='ffmpeg', fps=2)
    plt.show()


def main() -> None:
    """Main function to run Q-learning agent in Modified TSP environment."""
    num_targets = 10
    shuffle_time = 10
    num_episodes = 500  

    env = ModTSP(num_targets, shuffle_time=shuffle_time)
    agent = QLearningAgent(n_actions=num_targets)

    ep_rets = []  #  cumulative rewards
    avg_losses = []  #  average loss (TD error)

    visited_states_all = []  # For storing states visited in each episode
    episodic_rewards = {}
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        visited_states = []

        while not done:
            action = agent.choose_action(state)
            visited_states.append(action)  #  the state visited
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update_q_value(state, action, reward, next_state)

            total_reward += reward
            state = next_state

        visited_states_all.append(visited_states)
        ep_rets.append(total_reward)
        avg_losses.append(np.mean(agent.epsilon))
        if ep%50==0:
            episodic_rewards["Cummulative rewards after episode {}".format(ep)] = total_reward
    print(episodic_rewards)

    plt.plot(ep_rets, label='Cumulative Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Episode vs Cumulative Reward')
    plt.legend()
    plt.show()

    animate_solution(env, agent, visited_states_all[-1])


if __name__ == "__main__":
    main()
