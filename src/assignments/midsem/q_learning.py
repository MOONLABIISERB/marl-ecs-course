import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
from assignments.midsem.given import ModTSP

console = Console()


class QLearningAgent:
    """
    Q-Learning agent for the Modified Travelling Salesman Problem.

    Args:
        num_targets (int): Number of targets in the environment.
        num_actions (int): Number of possible actions.
        learning_rate (float): Learning rate for Q-value updates.
        discount_factor (float): Discount factor for future rewards.
        epsilon (float): Epsilon value for epsilon-greedy action selection.

    Attributes:
        q_table (Dict): Q-table storing state-action values.
        lr (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Epsilon for exploration.
    """

    def __init__(
        self,
        num_targets: int,
        num_actions: int,
        learning_rate: float = 0.0001,
        discount_factor: float = 0.9,
        epsilon: float = 0.0001,
    ) -> None:
        self.num_targets = num_targets
        self.q_table: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def discretize_state(self, state: np.ndarray) -> Tuple[int, Tuple[int, ...]]:
        """
        Discretize the continuous state space.

        Args:
            state (np.ndarray): The continuous state from the environment.

        Returns:
            Tuple[int, Tuple[int, ...]]: Discretized state (current location, visited targets).
        """
        current_location = int(state[0])
        visited_targets = tuple(state[1 : self.num_targets + 1].astype(int))
        return (current_location, visited_targets)

    def get_action(self, state: np.ndarray) -> int:
        """
        Get action using epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state.

        Returns:
            int: The chosen action.
        """
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.num_targets)

        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_targets)
        else:
            return int(np.argmax(self.q_table[discrete_state]))

    def update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        """
        Update Q-values based on the observed transition.

        Args:
            state (np.ndarray): Current state.
            action (int): Chosen action.
            reward (float): Received reward.
            next_state (np.ndarray): Next state.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.num_targets)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.num_targets)

        best_next_action = np.argmax(self.q_table[discrete_next_state])
        td_target = (
            reward + self.gamma * self.q_table[discrete_next_state][best_next_action]
        )
        td_error = td_target - self.q_table[discrete_state][action]
        self.q_table[discrete_state][action] += self.lr * td_error


def train_q_learning(
    env: ModTSP, num_episodes: int = 100000
) -> Tuple[List[float], QLearningAgent]:
    """
    Train the Q-Learning agent on the Modified TSP environment.

    Args:
        env (ModTSP): The Modified TSP environment.
        num_episodes (int): Number of training episodes.

    Returns:
        List[float]: List of episode rewards.
    """
    agent = QLearningAgent(env.num_targets, env.action_space.n, learning_rate=1e-6)
    episode_rewards = []

    with Progress() as progress:
        task = progress.add_task("[cyan]Training...", total=num_episodes)

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)

            if episode % 100 == 0:
                progress.update(
                    task,
                    advance=100,
                    description=f"[cyan]Training... Reward: {np.mean(episode_rewards[-100:]):.2f}",
                )

    return episode_rewards, agent


def plot_results(rewards: List[float]) -> None:
    """
    Plot the training results.

    Args:
        rewards (List[float]): List of episode rewards.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.6)
    plt.title("Q-learning Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, linestyle="--", alpha=0.7)

    window_size = 1000
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
    plt.plot(range(window_size - 1, len(rewards)), moving_avg, color="red", linewidth=2)

    plt.legend(["Episode Reward", f"{window_size}-Episode Moving Average"])
    plt.tight_layout()
    plt.show()


def demonstrate_policy(env: ModTSP, agent: QLearningAgent) -> None:
    """
    Demonstrate the learned policy.

    Args:
        env (ModTSP): The Modified TSP environment.
        agent (QLearningAgent): The trained Q-Learning agent.
    """
    state, _ = env.reset()
    done = False
    total_reward = 0
    trajectory = [env.loc]

    console.print(Panel("Demonstrating Learned Policy", style="bold green"))
    console.print(f"Starting Location: {env.loc}")

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        trajectory.append(env.loc)

        console.print(f"Move to: {env.loc}, Reward: {reward:.2f}")

        state = next_state

    console.print(f"\nTrajectory: {' -> '.join(map(str, trajectory))}")
    console.print(f"Total Reward: {total_reward:.2f}")

    # Visualize the trajectory
    locations = env.locations
    plt.figure(figsize=(10, 10))
    plt.scatter(locations[:, 0], locations[:, 1], c="blue", s=200)
    for i, loc in enumerate(locations):
        plt.annotate(
            f"{i}", (loc[0], loc[1]), xytext=(10, 10), textcoords="offset points"
        )

    for i in range(len(trajectory) - 1):
        start = locations[trajectory[i]]
        end = locations[trajectory[i + 1]]
        plt.arrow(
            start[0],
            start[1],
            end[0] - start[0],
            end[1] - start[1],
            color="red",
            head_width=0.3,
            head_length=0.3,
            length_includes_head=True,
        )

    plt.title("Agent Trajectory")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.show()


def main() -> None:
    """Main function to run the Q-Learning algorithm on the Modified TSP."""
    # Step 1: Initialize the environment
    console.print(Panel("Initializing Modified TSP Environment", style="bold green"))
    env = ModTSP(seed=69)

    # Step 2: Train the Q-Learning agent
    console.print(Panel("Starting Q-Learning Training", style="bold green"))
    rewards, agent = train_q_learning(env, num_episodes=500)
    print(len(list(agent.q_table.items())))

    # Step 3: Plot training results
    console.print(Panel("Plotting Training Results", style="bold green"))
    plot_results(rewards)

    # Step 4: Demonstrate learned policy
    console.print(Panel("Demonstrating Learned Policy", style="bold green"))
    # agent = QLearningAgent(env.num_targets, env.action_space.n)
    demonstrate_policy(env, agent)


if __name__ == "__main__":
    main()
