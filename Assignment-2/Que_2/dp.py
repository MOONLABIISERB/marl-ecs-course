import numpy as np
from tqdm import tqdm

class DynamicProgramming:
    def __init__(self, environment):
        self.env = environment
        self.all_states = self._generate_state_space()
        self.state_index_map = {state: idx for idx, state in enumerate(self.all_states)}
        self.total_states = len(self.all_states)
        self.total_actions = environment.action_space.n
        self.value_function = np.zeros(self.total_states)
        self.policy = np.zeros(self.total_states, dtype=int)

    def _generate_state_space(self):
        state_space = []
        for player_pos in [
            (i, j) for i in range(self.env.height) for j in range(self.env.width)
        ]:
            for box_pos in [
                (i, j) for i in range(self.env.height) for j in range(self.env.width)
            ]:
                for storage_pos in [
                    (i, j) for i in range(self.env.height) for j in range(self.env.width)
                ]:
                    if (
                        player_pos != box_pos
                        and box_pos != storage_pos
                        and player_pos != storage_pos
                    ):
                        state_space.append((player_pos, box_pos, storage_pos))
        return state_space

    def perform_value_iteration(self, gamma=0.9, convergence_threshold=1e-6, max_iter=1000):
        for _ in tqdm(range(max_iter), desc="Value Iteration"):
            max_diff = 0
            for idx, state in enumerate(self.all_states):
                old_value = self.value_function[idx]
                action_values = []
                for action in range(self.total_actions):
                    next_state, reward, done = self._calculate_next_state(state, action)
                    if next_state in self.state_index_map:
                        next_state_idx = self.state_index_map[next_state]
                        action_values.append(
                            reward + gamma * self.value_function[next_state_idx] * (not done)
                        )
                    else:
                        action_values.append(reward)
                self.value_function[idx] = max(action_values)
                max_diff = max(max_diff, abs(old_value - self.value_function[idx]))
            if max_diff < convergence_threshold:
                break

        self._update_policy(gamma)
        return self.policy

    def _calculate_next_state(self, state, action):
        self.env.reset()
        self.env.player_pos, self.env.box_pos, self.env.storage_pos = state
        self.env.grid[self.env.player_pos[0], self.env.player_pos[1]] = self.env.PLAYER
        self.env.grid[self.env.box_pos[0], self.env.box_pos[1]] = self.env.BOX
        self.env.grid[self.env.storage_pos[0], self.env.storage_pos[1]] = self.env.STORAGE

        next_state_obs, reward, done, _, _ = self.env.step(action)
        next_state = self.env.get_current_state()

        return next_state, reward, done

    def _update_policy(self, gamma):
        for idx, state in enumerate(self.all_states):
            action_values = []
            for action in range(self.total_actions):
                next_state, reward, done = self._calculate_next_state(state, action)
                if next_state in self.state_index_map:
                    next_state_idx = self.state_index_map[next_state]
                    action_values.append(
                        reward + gamma * self.value_function[next_state_idx] * (not done)
                    )
                else:
                    action_values.append(reward)
            self.policy[idx] = np.argmax(action_values)

    def select_action(self, state):
        if state in self.state_index_map:
            return self.policy[self.state_index_map[state]]
        return np.random.randint(self.total_actions)
