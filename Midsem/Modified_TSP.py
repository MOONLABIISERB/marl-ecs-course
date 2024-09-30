import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt  # Import for plotting
from typing import Dict, List, Optional, Tuple


class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits."""

    def __init__(self, num_targets: int = 10, max_area: int = 15, shuffle_time: int = 10, seed: int = 42) -> None:
        super().__init__()

        np.random.seed(seed)
        self.steps = 0
        self.episodes = 0
        self.shuffle_time = shuffle_time
        self.num_targets = num_targets

        self.max_steps = num_targets
        self.max_area = max_area

        self.locations = self._generate_points(self.num_targets)
        self.distances = self._calculate_distances(self.locations)

        # Initialize profits for each target
        self.initial_profits = np.arange(1, self.num_targets + 1, dtype=np.float32) * 10.0
        self.current_profits = self.initial_profits.copy()

        # Observation Space: {current loc, target flag - visited or not, current profits, distances, coordinates}
        self.obs_low = np.concatenate(
            [np.array([0], dtype=np.float32), np.zeros(self.num_targets, dtype=np.float32),
             np.zeros(self.num_targets, dtype=np.float32), np.zeros(self.num_targets, dtype=np.float32),
             np.zeros(2 * self.num_targets, dtype=np.float32)]
        )

        self.obs_high = np.concatenate(
            [np.array([self.num_targets], dtype=np.float32), np.ones(self.num_targets, dtype=np.float32),
             100 * np.ones(self.num_targets, dtype=np.float32),
             2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),
             self.max_area * np.ones(2 * self.num_targets, dtype=np.float32)]
        )

        # Action Space: {next_target}
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.num_targets)

        # Track total distance traveled (NEW)
        self.total_distance_traveled = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, None]]:
        """Reset the environment to the initial state."""
        self.steps = 0
        self.episodes += 1

        self.loc = 0
        self.visited_targets = np.zeros(self.num_targets)
        self.current_profits = self.initial_profits.copy()
        self.dist = self.distances[self.loc]

        # Reset total distance traveled to 0 at the beginning of each episode (NEW)
        self.total_distance_traveled = 0.0

        if self.episodes % self.shuffle_time == 0:
            np.random.shuffle(self.initial_profits)

        state = np.concatenate(
            (np.array([self.loc]), self.visited_targets, self.initial_profits, self.dist, self.locations.reshape(-1)),
            dtype=np.float32
        )
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        """Take an action (move to the next target)."""
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        # Update the total distance traveled (NEW)
        distance_traveled = self.distances[past_loc, next_loc]
        self.total_distance_traveled += distance_traveled

        # Reduce profits for all targets based on total distance traveled so far (MODIFIED)
        self.current_profits = self.initial_profits - self.total_distance_traveled

        # Get the reward based on the next location
        reward = self._get_rewards(next_loc)

        self.visited_targets[next_loc] = 1
        next_dist = self.distances[next_loc]
        terminated = bool(self.steps == self.max_steps)
        truncated = False

        next_state = np.concatenate(
            [np.array([next_loc]), self.visited_targets, self.current_profits, next_dist, self.locations.reshape(-1)],
            dtype=np.float32
        )

        self.loc, self.dist = next_loc, next_dist
        return next_state, reward, terminated, truncated, {}

    def _generate_points(self, num_points: int) -> np.ndarray:
        """Generate random 2D points representing target locations."""
        return np.random.uniform(low=0, high=self.max_area, size=(num_points, 2)).astype(np.float32)

    def _calculate_distances(self, locations: np.ndarray) -> np.ndarray:
        """Calculate the distance matrix between all target locations."""
        n = len(locations)
        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, next_loc: int) -> float:
        """Calculate the reward based on the distance traveled, however if a target gets visited again then it incurs a high penalty."""
        reward = self.current_profits[next_loc] if not self.visited_targets[next_loc] else -1e4
        return float(reward)


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions  # Number of possible actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor for future rewards
        self.q_table = {}   # Initialize the Q-table as a dictionary
        self.td_errors = []  # Track TD errors for plotting average loss

    def get_q_value(self, state, action):
        """Retrieve the Q-value for a specific state-action pair."""
        state_str = str(state)  # Convert state to string for Q-table indexing
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(self.n_actions)  # Initialize Q-values to 0
        return self.q_table[state_str][action]

    def choose_action(self, state):
        """Choose the best action based on the current Q-values (greedy policy)."""
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(self.n_actions)  # Initialize Q-values to 0 if new state
        return np.argmax(self.q_table[state_str])  # Always choose the action with the highest Q-value

    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value for the given state-action pair."""
        state_str = str(state)
        next_state_str = str(next_state)

        # Initialize next state in Q-table if not present
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(self.n_actions)

        # Q-learning update rule: Q(s, a) = Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
        best_next_action = np.argmax(self.q_table[next_state_str])  # Greedy selection of next action
        td_target = reward + self.gamma * self.q_table[next_state_str][best_next_action]  # TD target
        td_error = td_target - self.q_table[state_str][action]  # TD error

        # Track TD error (NEW)
        self.td_errors.append(abs(td_error))

        # Update Q-value
        self.q_table[state_str][action] += self.alpha * td_error


def main() -> None:
    """Main function to run Q-learning agent in Modified TSP environment."""
    num_targets = 10
    shuffle_time = 2000  # Shuffle profits after every 10 episodes
    num_episodes = 9999

    env = ModTSP(num_targets, shuffle_time=shuffle_time)
    agent = QLearningAgent(n_actions=num_targets)

    ep_rets = []  # Track cumulative rewards
    avg_distances = []  # Track average distance traveled per episode
    avg_losses = []  # Track average loss (TD error)

    for ep in range(num_episodes):  # Run for 1000 episodes
        state, _ = env.reset()  # Reset the environment at the start of each episode
        done = False
        total_reward = 0
        total_distance = 0

        while not done:
            action = agent.choose_action(state)  # Always choose the best action (greedy)
            next_state, reward, terminated, truncated, _ = env.step(action)  # Step in the environment
            done = terminated or truncated

            agent.update_q_value(state, action, reward, next_state)  # Update Q-values

            total_reward += reward
            total_distance = env.total_distance_traveled  # Track total distance per episode
            state = next_state  # Move to the next state

        ep_rets.append(total_reward)
        avg_distances.append(total_distance)  # Append total distance for the episode
        avg_losses.append(np.mean(agent.td_errors))  # Calculate and track average loss

        
        print(f"Episode {ep} / {num_episodes}: Total Reward Collected: {total_reward}, Total Distance: {total_distance:.2f}")

    print(f"Average reward over {num_episodes} episodes: {np.mean(ep_rets)}")

    # Plot cumulative reward
    plt.figure(figsize=(12, 4))
    
    plt.plot(ep_rets, label='Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward per Episode')
    plt.legend()
    plt.show()

    # Plot average distance traveled
    plt.figure(figsize=(12, 4))
    plt.plot(avg_distances, label='Average Distance Traveled', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Average Distance')
    plt.title('Average Distance per Episode')
    plt.legend()
    plt.show()

    # Plot average TD error (loss)
    plt.figure(figsize=(12, 4))
    plt.plot(avg_losses, label='Average TD Error (Loss)', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Average Loss (TD Error) per Episode')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
