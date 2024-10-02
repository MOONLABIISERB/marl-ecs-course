from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt
import random



class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits.

    The agent navigates a set of targets based on precomputed distances. It aims to visit
    all targets so maximize profits. The profits decay with time.
    """

    def __init__(
        self,
        num_targets: int = 10,
        max_area: int = 15,
        shuffle_time: int = 10,
        seed: int = 42,
    ) -> None:
        """Initialize the TSP environment.

        Args:
            num_targets (int): No. of targets the agent needs to visit.
            max_area (int): Max. Square area where the targets are defined.
            shuffle_time (int): No. of episodes after which the profits ar to be shuffled.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()

        np.random.seed(seed)

        self.steps: int = 0
        self.episodes: int = 0

        self.shuffle_time: int = shuffle_time
        self.num_targets: int = num_targets

        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations: npt.NDArray[np.float32] = self._generate_points(self.num_targets)
        self.distances: npt.NDArray[np.float32] = self._calculate_distances(self.locations)

        # Initialize profits for each target
        self.initial_profits: npt.NDArray[np.float32] = np.arange(1, self.num_targets + 1, dtype=np.float32) * 10.0
        self.current_profits: npt.NDArray[np.float32] = self.initial_profits.copy()

        # Observation Space : {current loc (loc), target flag - visited or not, current profits, dist_array (distances), coordintates (locations)}
        self.obs_low = np.concatenate(
            [
                np.array([0], dtype=np.float32),  # Current location
                np.zeros(self.num_targets, dtype=np.float32),  # Check if targets were visited or not
                np.zeros(self.num_targets, dtype=np.float32),  # Array of all current profits values
                np.zeros(self.num_targets, dtype=np.float32),  # Distance to each target from current location
                np.zeros(2 * self.num_targets, dtype=np.float32),  # Cooridinates of all targets
            ]
        )

        self.obs_high = np.concatenate(
            [
                np.array([self.num_targets], dtype=np.float32),  # Current location
                np.ones(self.num_targets, dtype=np.float32),  # Check if targets were visited or not
                100 * np.ones(self.num_targets, dtype=np.float32),  # Array of all current profits values
                2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),  # Distance to each target from current location
                self.max_area * np.ones(2 * self.num_targets, dtype=np.float32),  # Cooridinates of all targets
            ]
        )

        # Action Space : {next_target}
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.num_targets)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, None]]:
        """Reset the environment to the initial state.

        Args:
            seed (Optional[int], optional): Seed to reset the environment. Defaults to None.
            options (Optional[dict], optional): Additional reset options. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, None]]: The initial state of the environment and an empty info dictionary.
        """
        self.steps: int = 0
        self.episodes += 1

        self.loc: int = 0
        self.visited_targets: npt.NDArray[np.float32] = np.zeros(self.num_targets)
        self.current_profits = self.initial_profits.copy()
        self.dist: List = self.distances[self.loc]

        if self.shuffle_time % self.episodes == 0:
            np.random.shuffle(self.initial_profits)

        state = np.concatenate(
            (
                np.array([self.loc]),
                self.visited_targets,
                self.initial_profits,
                np.array(self.dist),
                np.array(self.locations).reshape(-1),
            ),
            dtype=np.float32,
        )
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        """Take an action (move to the next target).

        Args:
            action (int): The index of the next target to move to.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
                - The new state of the environment.
                - The reward for the action.
                - A boolean indicating whether the episode has terminated.
                - A boolean indicating if the episode is truncated.
                - An empty info dictionary.
        """
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        self.current_profits -= self.distances[past_loc, next_loc]
        reward = self._get_rewards(next_loc)

        self.visited_targets[next_loc] = 1

        next_dist = self.distances[next_loc]
        terminated = bool(self.steps == self.max_steps)
        truncated = False

        next_state = np.concatenate(
            [
                np.array([next_loc]),
                self.visited_targets,
                self.current_profits,
                next_dist,
                np.array(self.locations).reshape(-1),
            ],
            dtype=np.float32,
        )

        self.loc, self.dist = next_loc, next_dist
        return (next_state, reward, terminated, truncated, {})

    def _generate_points(self, num_points: int) -> npt.NDArray[np.float32]:
        """Generate random 2D points representing target locations.

        Args:
            num_points (int): Number of points to generate.

        Returns:
            np.ndarray: Array of 2D coordinates for each target.
        """
        return np.random.uniform(low=0, high=self.max_area, size=(num_points, 2)).astype(np.float32)

    def _calculate_distances(self, locations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calculate the distance matrix between all target locations.

        Args:
            locations: List of 2D target locations.

        Returns:
            np.ndarray: Matrix of pairwise distances between targets.
        """
        n = len(locations)

        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, next_loc: int) -> float:
        """Calculate the reward based on the distance traveled, however if a target gets visited again then it incurs a high penalty.

        Args:
            next_loc (int): Next location of the agent.

        Returns:
            float: Reward based on the travel distance between past and next locations, or negative reward if repeats visit.
        """
        reward = self.current_profits[next_loc] if not self.visited_targets[next_loc] else -1e4
        return float(reward)
    



class QLearning:
    """
    Q-Learning Agent for the Modified TSP Environment.
    """
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 500,
        num_targets: int = 10,
    ):
        """
        Initialize Q-learning Agent.

        Args:
        - alpha (float): Learning rate.
        - gamma (float): Discount factor.
        - epsilon_start (float): Initial epsilon value.
        - epsilon_end (float): Final epsilon value.
        - epsilon_decay_steps (int): Number of steps to decay epsilon.
        - num_targets (int): Number of targets in the environment.

        Returns:
        - None
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps # Linear decay.
        self.num_targets = num_targets


        self.Q_table = {}
        self.td_errors = []
        random.seed(42)



    def _state_to_string(self, state: np.ndarray) -> str:
        """Convert the state to a string for dictionary indexing.

        Args:
        - state (np.ndarray): Current state of the environment.

        Returns:
        - str: String representation of the state.
        """
        return str(state.tolist())



    def select_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
        - state (np.ndarray): Current state of the environment.

        Returns:
        - int: Action to take in the current state.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        state_rep = self._state_to_string(state)

        if state_rep not in self.Q_table:
            self.Q_table[state_rep] = np.zeros(self.num_targets)  # Initialize Q-values to 0 if new state

        if random.random() < self.epsilon:
            return random.randrange(self.num_targets)  # Explore
        else:
            return np.argmax(self.Q_table[state_rep])  # Exploit



    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        """Update the Q-value table using the Q-learning update rule.

        Args:
        - state (np.ndarray): Current state of the environment.
        - action (int): Action taken in the current state.
        - reward (float): Reward received after taking the action.
        - next_state (np.ndarray): Next state of the environment.

        Returns:
        - None
        """
        state_rep = self._state_to_string(state)
        next_state_rep = self._state_to_string(next_state)

        if state_rep not in self.Q_table:
            self.Q_table[state_rep] = np.zeros(self.num_targets)

        if next_state_rep not in self.Q_table:
            self.Q_table[next_state_rep] = np.zeros(self.num_targets)

        # Update Rule
        best_next_action = np.argmax(self.Q_table[next_state_rep])
        td_target = reward + self.gamma * self.Q_table[next_state_rep][best_next_action]
        td_error = td_target - self.Q_table[state_rep][action]

        # Update Q-value
        self.Q_table[state_rep][action] += self.alpha * td_error
        self.td_errors.append(abs(td_error)) # Store TD error for plotting
    


    def print_Q_table(self) -> None:
        """Print the Q-value table for all states.

        Returns:
        - None
        """
        print("Q-Value Table:")
        for state in self.Q_table:
            print(f"Q-Values: {self.Q_table}")

    
    def save_Q_table(self) -> None:
        """Save the Q-value table to a pickle file.

        Returns:
        - None
        """
        import pickle
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.Q_table, f)
        


def plot_cum_rew(ep_returns: List[float], 
                 window_size: int, 
                 num_episodes: int, 
                 moving_avg: List[float],
                 epsilon_start: float,
                 epsilon_end: float,
                 epsilon_decay_steps: int) -> None:
    """Plot the learning curve for the Q-Learning algorithm.

    Args:
    - ep_rets (List[float]): List of total rewards per episode.
    - window_size (int): Size of the moving average window.
    - num_episodes (int): Number of episodes run.
    - moving_avg (List[float]): List of moving average rewards.
    - epsilon_start (float): Initial epsilon value.
    - epsilon_end (float): Final epsilon value.
    - epsilon_decay_steps (int): Number of steps to decay epsilon.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(ep_returns, label='Total Reward per Episode', alpha=0.6)
    plt.plot(range(window_size - 1, num_episodes), moving_avg, label='Moving Average (window size = 10)', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning on ModTSP: Episode vs Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'q_learning_{num_episodes}_{epsilon_start}_{epsilon_end}_{epsilon_decay_steps}.png')
    plt.show()



def plot_loss(loss: List[float], 
              num_episodes: int, 
              epsilon_start: float, 
              epsilon_end: float,
              epsilon_decay_steps: int) -> None:
    """Plot the loss curve for the Q-Learning algorithm.

    Args:
    - loss (List[float]): List of TD errors per episode.
    - num_episodes (int): Number of episodes run.
    - epsilon_start (float): Initial epsilon value.
    - epsilon_end (float): Final epsilon value.
    - epsilon_decay_steps (int): Number of steps to decay epsilon.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(loss, label='TD error per Episode', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('TD Error')
    plt.title('Q-Learning on ModTSP: Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'q_learning_loss_{num_episodes}_{epsilon_start}_{epsilon_end}_{epsilon_decay_steps}.png')
    plt.show()




def visualize_optimal_path(env: ModTSP, path: List[int]) -> None:
    """
    Visualize the optimal path taken by the agent in the TSP environment using one-sided arrows (ax.arrow).

    Args:
    - env (ModTSP): The TSP environment containing the target locations.
    - path (List[int]): The list of target indices representing the path taken by the agent.

    Returns:
    - None
    """
    locations = env.locations
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the targets
    ax.scatter(locations[:, 0], locations[:, 1], c='blue', label="Targets", s=100)

    # Labelling each target
    for i, (x, y) in enumerate(locations):
        ax.text(x, y, f'{i}', fontsize=12, ha='right')
        

    path_locations = locations[path]
    for i in range(len(path) - 1):
        start_x, start_y = path_locations[i]
        end_x, end_y = path_locations[i + 1]
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Arrow to next point
        ax.arrow(start_x, start_y, dx, dy, head_width=0.3, head_length=0.5, fc='red', ec='red', alpha=0.7, length_includes_head=True)

    # Highlight the start location
    start_loc = locations[path[0]]
    ax.scatter(start_loc[0], start_loc[1], c='green', label="Start", s=200, edgecolor='black', zorder=5)

    ax.set_title("Path Visualization")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.grid(True)
    
    plt.savefig('optimal_path_ax_arrow.png')
    plt.show()






def main() -> None:
    """Main function to train the Q-Learning agent and visualize the optimal path."""
    num_targets = 10
    env = ModTSP(num_targets)
    state, _ = env.reset()

    # Hyperparameters
    num_episodes = 64999
    alpha = 0.03  
    gamma = 0.99
    epsilon_start = 0.5
    epsilon_end = 0.01
    epsilon_decay_steps = 100000

    # Initialize Agent
    agent = QLearning(
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        num_targets=num_targets,
    )

    ep_returns = []
    loss = []

    optimal_path = []  

    for ep in range(1, num_episodes + 1):
        total_reward = 0
        state, _ = env.reset()
        episode_path = []  

        for _ in range(env.max_steps):
            action = agent.select_action(state)
            episode_path.append(action)  # Add action (target index) to the episode path

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Update agent and moving to next state
            agent.update(state, action, reward, next_state)
            state = next_state

            if done:
                break

        ep_returns.append(total_reward)
        loss.append(np.mean(agent.td_errors))

        
        if ep == num_episodes:
            optimal_path = episode_path

        if ep % 10 == 0:
            print(f"Episode {ep} : Total Reward = {total_reward:.2f} | Epsilon = {agent.epsilon:.3f}")

    print(f"Episodes: {num_episodes} | Average Return: {np.mean(ep_returns):.2f}")
    print(f"Episodes: {num_episodes} | Average TD error: {np.mean(loss):.2f}")

    # agent.print_Q_table()
    agent.save_Q_table()

    # Plot the learning curve with moving average
    window_size = 10
    moving_avg = np.convolve(ep_returns, np.ones(window_size) / window_size, mode='valid')

    plot_cum_rew(ep_returns, window_size, num_episodes, moving_avg, epsilon_start, epsilon_end, epsilon_decay_steps)
    plot_loss(loss, num_episodes, epsilon_start, epsilon_end, epsilon_decay_steps)

    # Visualize the optimal path
    print(f"Path: {optimal_path}")
    visualize_optimal_path(env, optimal_path)


if __name__ == "__main__":
    main()
