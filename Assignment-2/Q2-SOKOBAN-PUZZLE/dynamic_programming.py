# dynamic_programming.py

import numpy as np
from tqdm import tqdm


class DynamicProgrammingAgent:
    """
    Agent that uses Dynamic Programming for policy computation.
    """

    def __init__(self, env):
        self.env = env
        self.states = self._enumerate_states()
        self.state_indices = {state: idx for idx, state in enumerate(self.states)}
        self.num_states = len(self.states)
        self.num_actions = env.action_space.n
        self.value_function = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)

    def _enumerate_states(self):
        # Generate all possible states
        states = []
        positions = [
            (i, j) for i in range(self.env.grid_height) for j in range(self.env.grid_width)
        ]
        for player_pos in positions:
            for box_pos in positions:
                for target_pos in positions:
                    if player_pos != box_pos and box_pos != target_pos and player_pos != target_pos:
                        states.append((player_pos, box_pos, target_pos))
        return states

    def value_iteration(self, discount_factor=0.99, threshold=1e-6, max_iterations=1000):
        # Perform value iteration
        for _ in tqdm(range(max_iterations), desc="Value Iteration Progress"):
            delta = 0
            for idx, state in enumerate(self.states):
                v = self.value_function[idx]
                action_values = []
                for action in range(self.num_actions):
                    next_state, reward, done = self._simulate_step(state, action)
                    if next_state in self.state_indices:
                        next_idx = self.state_indices[next_state]
                        action_value = reward + discount_factor * self.value_function[next_idx] * (not done)
                    else:
                        action_value = reward
                    action_values.append(action_value)
                self.value_function[idx] = max(action_values)
                delta = max(delta, abs(v - self.value_function[idx]))
            if delta < threshold:
                break
        self._derive_policy(discount_factor)

    def _simulate_step(self, state, action):
        # Simulate the environment step without modifying the actual environment
        self.env.reset()
        self.env.player_position, self.env.box_position, self.env.target_position = state
        self.env.grid[self.env.player_position[0], self.env.player_position[1]] = self.env.PLAYER
        self.env.grid[self.env.box_position[0], self.env.box_position[1]] = self.env.BOX
        self.env.grid[self.env.target_position[0], self.env.target_position[1]] = self.env.TARGET

        _, reward, done, _, _ = self.env.step(action)
        next_state = self.env.get_state()

        return next_state, reward, done

    def _derive_policy(self, discount_factor):
        # Extract policy from the computed value function
        for idx, state in enumerate(self.states):
            action_values = []
            for action in range(self.num_actions):
                next_state, reward, done = self._simulate_step(state, action)
                if next_state in self.state_indices:
                    next_idx = self.state_indices[next_state]
                    action_value = reward + discount_factor * self.value_function[next_idx] * (not done)
                else:
                    action_value = reward
                action_values.append(action_value)
            self.policy[idx] = np.argmax(action_values)

    def select_action(self, state):
        # Select action based on the derived policy
        if state in self.state_indices:
            idx = self.state_indices[state]
            return self.policy[idx]
        else:
            return self.env.action_space.sample()
