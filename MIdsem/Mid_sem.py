# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict
from typing import Dict, Tuple

# Define the ModTSP environment
class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits."""

    def __init__(self, num_targets: int = 10, max_area: int = 15, shuffle_time: int = 10, seed: int = 42):
        super().__init__()
        np.random.seed(seed)

        self.steps: int = 0
        self.episodes: int = 0

        self.shuffle_time: int = shuffle_time
        self.num_targets: int = num_targets

        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations = self._generate_points(self.num_targets)
        self.distances = self._calculate_distances(self.locations)

        self.initial_profits = np.arange(1, self.num_targets + 1, dtype=np.float32) * 10.0
        self.current_profits = self.initial_profits.copy()

        self.action_space = gym.spaces.Discrete(self.num_targets)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.num_targets * 2 + 2,), dtype=np.float32)

    def reset(self) -> Tuple[np.ndarray, Dict[str, None]]:
        self.steps = 0
        self.episodes += 1
        self.loc = 0
        self.visited_targets = np.zeros(self.num_targets)
        self.current_profits = self.initial_profits.copy()

        if self.shuffle_time % self.episodes == 0:
            np.random.shuffle(self.initial_profits)

        state = np.concatenate([
            np.array([self.loc]),
            self.visited_targets,
            self.current_profits,
            np.zeros(self.num_targets), 
            self.locations.flatten()
        ], dtype=np.float32)

        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        self.current_profits -= self.distances[past_loc, next_loc]
        reward = self._get_rewards(next_loc)

        self.visited_targets[next_loc] = 1
        terminated = self.steps >= self.max_steps

        next_state = np.concatenate([
            np.array([next_loc]),
            self.visited_targets,
            self.current_profits,
            self.distances[next_loc],
            self.locations.flatten()
        ], dtype=np.float32)

        self.loc = next_loc
        return next_state, reward, terminated, False, {}

    def _generate_points(self, num_points: int) -> np.ndarray:
        return np.random.uniform(low=0, high=self.max_area, size=(num_points, 2)).astype(np.float32)

    def _calculate_distances(self, locations: np.ndarray) -> np.ndarray:
        n = len(locations)
        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, next_loc: int) -> float:
        return self.current_profits[next_loc] if not self.visited_targets[next_loc] else -1e4


class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, epsilon_decay=0.99):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Explore 
        else:
            return np.argmax(self.q_table[state])  # Exploit state init

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.learning_rate * (td_target - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

# Main function 
def main():
    num_targets = 10
    env = ModTSP(num_targets)  

    # Initialize 
    agent = QLearningAgent(env.action_space)

    num_episodes = 2500  # Number of training episodes
    cumulative_rewards = []  # Store total rewards for each episode

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = tuple(state.astype(int))

        total_reward = 0
        terminated = False

        while not terminated:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_state.astype(int))  

           
            agent.update(state, action, reward, next_state)

            
            state = next_state
            total_reward += reward

        # Decay epsilon for expolration reduction
        agent.decay_epsilon()

        
        cumulative_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Plot of training data
    plt.plot(cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Episode vs Cumulative Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
if __name__ == "__main__":
    main()

