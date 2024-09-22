import numpy as np
import gymnasium as gym

class SokobanEnv(gym.Env):
    """Grid-World Sokoban Environment."""

    def __init__(self):
        self.grid = np.zeros((6, 7), dtype=int)
        self.agent_pos = [0, 0]  # Initial position of the agent
        self.boxes = [[2, 2], [3, 4]]  # Initial positions of boxes
        self.storage = [[1, 1], [4, 5]]  # Storage locations
        self.max_steps = 100
        self.steps = 0

        # Action space: 4 actions (UP, DOWN, LEFT, RIGHT)
        self.action_space = gym.spaces.Discrete(4)
        # Observation space: (row, column) for agent position
        self.observation_space = gym.spaces.MultiDiscrete([6, 7])

    def reset(self):
        """Resets the environment to the initial state."""
        self.agent_pos = [0, 0]
        self.steps = 0
        return self.agent_pos

    def step(self, action):
        """Moves the agent based on the action taken and returns the new state."""
        self.steps += 1

        # Move the agent based on action: UP, DOWN, LEFT, RIGHT
        if action == 0:  # UP
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # DOWN
            self.agent_pos[0] = min(5, self.agent_pos[0] + 1)
        elif action == 2:  # LEFT
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # RIGHT
            self.agent_pos[1] = min(6, self.agent_pos[1] + 1)

        # Determine if the game is done (either by reaching max steps or solving the puzzle)
        done = self.steps >= self.max_steps
        reward = -1  # Penalize for each step taken

        # If the agent reaches a storage location, give zero reward
        if self.agent_pos in self.storage:
            reward = 0

        return self.agent_pos, reward, done, {}

    def render(self):
        """Displays the current state of the environment."""
        grid_copy = self.grid.copy()
        grid_copy[tuple(self.agent_pos)] = 1  # Mark agent position
        for box in self.boxes:
            grid_copy[tuple(box)] = 2  # Mark box positions
        for storage in self.storage:
            grid_copy[tuple(storage)] = 3  # Mark storage locations

        print(grid_copy)

def value_iteration_sokoban(env, gamma=0.99, theta=1e-5):
    """Implements Value Iteration for the Sokoban environment."""
    state_space = [(i, j) for i in range(6) for j in range(7)]
    V = {state: 0 for state in state_space}
    policy = {state: 0 for state in state_space}

    while True:
        delta = 0
        for state in state_space:
            v = V[state]
            action_values = []
            for action in range(env.action_space.n):
                env.agent_pos = list(state)  # Set the agent position to the current state
                next_state, reward, done, _ = env.step(action)
                next_state = tuple(next_state)
                action_values.append(reward + gamma * V[next_state])
            V[state] = max(action_values)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break

    for state in state_space:
        action_values = []
        for action in range(env.action_space.n):
            env.agent_pos = list(state)
            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)
            action_values.append(reward + gamma * V[next_state])
        policy[state] = np.argmax(action_values)

    return policy, V

def monte_carlo_first_visit(env, episodes=1000, gamma=0.99):
    # Use a dictionary instead of a numpy array for V and returns to handle state tuples
    state_space = [(i, j) for i in range(env.num_targets)]
    returns = {state: [] for state in range(env.num_targets)}
    V = {state: 0 for state in range(env.num_targets)}

    for ep in range(episodes):
        episode = []
        state, _ = env.reset()  # Reset the environment to start
        state = int(state[0])  # Cast the state to an integer index for tracking
        done = False

        # Play an episode and store the sequence of states, actions, rewards
        while not done:
            action = env.action_space.sample()  # Random action (Exploring Starts)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = int(next_state[0])  # Ensure next_state is an integer
            episode.append((state, action, reward))
            state = next_state

        # Compute returns
        G = 0
        visited_states = set()
        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            if state not in visited_states:
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                visited_states.add(state)

    return V


if __name__ == "__main__":
    num_targets = 6
    env = TSP(num_targets)

    # Running Value Iteration
    policy, V = value_iteration(env)
    print("Value Iteration Policy:", policy)
    print("Value Iteration Values:", V)

    # Running Monte Carlo First Visit
    V_mc = monte_carlo_first_visit(env)
    print("Monte Carlo First Visit Values:", V_mc)