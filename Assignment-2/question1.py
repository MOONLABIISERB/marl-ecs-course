import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple

class TSP(gym.Env):
    """Traveling Salesman Problem (TSP) RL environment."""

    def __init__(self, num_targets: int, max_area: int = 30, seed: int = None) -> None:
        super().__init__()
        if seed is not None:
            np.random.seed(seed=seed)

        self.num_targets: int = num_targets
        self.max_steps: int = num_targets
        self.max_area: int = max_area
        self.steps: int = 0  # Initialize steps

        self.locations: np.ndarray = self._generate_points(self.num_targets)
        self.distances: np.ndarray = self._calculate_distances(self.locations)

        self.obs_low = np.concatenate(
            [np.array([0], dtype=np.float32), np.zeros(self.num_targets, dtype=np.float32), np.zeros(2 * self.num_targets, dtype=np.float32)]
        )
        self.obs_high = np.concatenate(
            [np.array([self.num_targets], dtype=np.float32), 2 * self.max_area * np.ones(self.num_targets, dtype=np.float32), self.max_area * np.ones(2 * self.num_targets, dtype=np.float32)]
        )

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.num_targets)

        # Initialize the agent's location and distances
        self.loc = 0
        self.visited_targets = []
        self.dist = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, None]]:
        """Resets the environment to the initial state."""
        self.steps = 0
        self.loc = 0  # Reset the agent to the initial location
        self.visited_targets = []
        self.dist = self.distances[self.loc]

        state = np.concatenate((np.array([self.loc]), np.array(self.dist), np.array(self.locations).reshape(-1)), dtype=np.float32)
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        """Moves to the next target and returns the new state, reward, and other info."""
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        reward = self._get_rewards(past_loc, next_loc)
        self.visited_targets.append(next_loc)

        next_dist = self.distances[next_loc]
        terminated = bool(self.steps == self.max_steps)
        truncated = False

        next_state = np.concatenate([np.array([next_loc]), next_dist, np.array(self.locations).reshape(-1)], dtype=np.float32)
        self.loc, self.dist = next_loc, next_dist

        return next_state, reward, terminated, truncated, {}

    def _generate_points(self, num_points: int) -> np.ndarray:
        """Generates random 2D points for the targets."""
        points = []
        while len(points) < num_points:
            x = np.random.random() * self.max_area
            y = np.random.random() * self.max_area
            if [x, y] not in points:
                points.append([x, y])
        return np.array(points)

    def _calculate_distances(self, locations: List) -> np.ndarray:
        """Calculates distances between target locations."""
        n = len(locations)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, past_loc: int, next_loc: int) -> float:
        if next_loc not in self.visited_targets:
        # Negative reward for the distance travelled
            reward = -self.distances[past_loc][next_loc]
        else:
        # Penalize for revisiting, but use a smaller penalty to avoid extreme values
            reward = -100  # You can change this to a smaller penalty
        return reward

def value_iteration(env, gamma=0.99, theta=1e-5):
    V = np.zeros(env.num_targets)
    policy = np.zeros(env.num_targets, dtype=int)

    while True:
        delta = 0
        for state in range(env.num_targets):
            v = V[state]
            action_values = []
            for action in range(env.num_targets):
                env.loc = state  # Set the current state (loc) in the environment
                next_state, reward, _, _, _ = env.step(action)
                
                # Cast next_state[0] to an integer to avoid the indexing issue
                next_state_index = int(next_state[0])
                
                action_values.append(reward + gamma * V[next_state_index])
            V[state] = max(action_values)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break

    # Create policy based on the best action for each state
    for state in range(env.num_targets):
        action_values = []
        for action in range(env.num_targets):
            env.loc = state  # Set the current state (loc) in the environment
            next_state, reward, _, _, _ = env.step(action)
            
            # Again, cast next_state[0] to an integer
            next_state_index = int(next_state[0])
            
            action_values.append(reward + gamma * V[next_state_index])
        policy[state] = np.argmax(action_values)

    return policy, V

def monte_carlo_first_visit(env, episodes=1000, gamma=0.99):
    returns = {state: [] for state in range(env.num_targets)}
    V = np.zeros(env.num_targets)

    for ep in range(episodes):
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state[0], action, reward))
            state = next_state

        G = 0
        visited_states = set()
        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            if state not in visited_states:
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                visited_states.add(state)

    return V

# Running TSP with Value Iteration and Monte Carlo First-Visit
if __name__ == "__main__":
    num_targets = 6
    env = TSP(num_targets)
    
    # Value Iteration
    policy, V = value_iteration(env)
    print("Value Iteration Policy:", policy)
    print("Value Iteration Values:", V)

    # Monte Carlo First Visit
    V_mc = monte_carlo_first_visit(env)
    print("Monte Carlo First Visit Values:", V_mc)
