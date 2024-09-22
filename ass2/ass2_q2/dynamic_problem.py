import numpy as np
from tqdm import tqdm

class DpSolver:
    """Dynamic Programming Solver for the environment using value iteration."""

    def __init__(self, env):
        self.env = env
        self.states_list = self._enumerate_states()
        self.state_index_map = {state: idx for idx, state in enumerate(self.states_list)}
        self.total_states = len(self.states_list)
        self.total_actions = env.action_space.n
        self.value_table = np.zeros(self.total_states)
        self.policy_table = np.zeros(self.total_states, dtype=int)

    def _enumerate_states(self):
        """Generate all possible combinations of player, box, and storage positions."""
        all_states = []
        for player_pos in [
            (x, y) for x in range(self.env.height) for y in range(self.env.width)
        ]:
            for box_pos in [
                (x, y) for x in range(self.env.height) for y in range(self.env.width)
            ]:
                for storage_pos in [
                    (x, y) for x in range(self.env.height) for y in range(self.env.width)
                ]:
                    # Ensure positions are distinct
                    if (
                        player_pos != box_pos
                        and box_pos != storage_pos
                        and player_pos != storage_pos
                    ):
                        all_states.append((player_pos, box_pos, storage_pos))
        return all_states

    def apply_value_iteration(self, gamma=0.99, tolerance=1e-8, max_iter=1000):
        """Perform value iteration to compute optimal value function and policy."""
        for _ in tqdm(range(max_iter), desc="Value Iteration Progress"):
            max_delta = 0
            for s_idx, state in enumerate(self.states_list):
                current_value = self.value_table[s_idx]
                action_values = []
                for action in range(self.total_actions):
                    next_state, reward, is_done = self._transition(state, action)
                    if next_state in self.state_index_map:
                        next_s_idx = self.state_index_map[next_state]
                        action_values.append(
                            reward + gamma * self.value_table[next_s_idx] * (not is_done)
                        )
                    else:
                        action_values.append(reward)
                self.value_table[s_idx] = max(action_values)
                max_delta = max(max_delta, abs(current_value - self.value_table[s_idx]))

            if max_delta < tolerance:
                break

        self._derive_policy(gamma)
        return self.policy_table

    def _transition(self, state, action):
        """Simulate taking an action and return the next state, reward, and done flag."""
        self.env.reset()
        self.env.player_pos, self.env.box_pos, self.env.storage_pos = state
        self.env.grid[self.env.player_pos[0], self.env.player_pos[1]] = self.env.PLAYER
        self.env.grid[self.env.box_pos[0], self.env.box_pos[1]] = self.env.BOX
        self.env.grid[self.env.storage_pos[0], self.env.storage_pos[1]] = self.env.STORAGE

        next_obs, reward, done, _, _ = self.env.step(action)
        next_state = self.env.get_state()

        return next_state, reward, done

    def _derive_policy(self, gamma):
        """Extract the optimal policy based on the computed value table."""
        for s_idx, state in enumerate(self.states_list):
            action_values = []
            for action in range(self.total_actions):
                next_state, reward, is_done = self._transition(state, action)
                if next_state in self.state_index_map:
                    next_s_idx = self.state_index_map[next_state]
                    action_values.append(
                        reward + gamma * self.value_table[next_s_idx] * (not is_done)
                    )
                else:
                    action_values.append(reward)
            self.policy_table[s_idx] = np.argmax(action_values)

    def select_action(self, state):
        """Select the best action based on the policy for a given state."""
        if state in self.state_index_map:
            return self.policy_table[self.state_index_map[state]]
        return np.random.randint(self.total_actions)  # Default to a random action if state not in policy
