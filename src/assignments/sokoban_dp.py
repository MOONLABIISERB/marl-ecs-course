import numpy as np
from tqdm import tqdm


class DynamicProgramming:
    def __init__(self, env):
        self.env = env
        self.states = self._generate_all_states()
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
        self.num_states = len(self.states)
        self.num_actions = env.action_space.n
        self.value_function = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)

    def _generate_all_states(self):
        states = []
        for player_pos in [
            (i, j) for i in range(self.env.height) for j in range(self.env.width)
        ]:
            for box_pos in [
                (i, j) for i in range(self.env.height) for j in range(self.env.width)
            ]:
                for storage_pos in [
                    (i, j)
                    for i in range(self.env.height)
                    for j in range(self.env.width)
                ]:
                    if (
                        player_pos != box_pos
                        and box_pos != storage_pos
                        and player_pos != storage_pos
                    ):
                        states.append((player_pos, box_pos, storage_pos))
        return states

    def value_iteration(self, gamma=0.99, theta=1e-8, max_iterations=1000):
        for _ in tqdm(range(max_iterations), desc="Value Iteration"):
            delta = 0
            for s, state in enumerate(self.states):
                v = self.value_function[s]
                values = []
                for a in range(self.num_actions):
                    next_state, reward, done = self._get_next_state_reward(state, a)
                    if next_state in self.state_to_index:
                        next_s = self.state_to_index[next_state]
                        values.append(
                            reward + gamma * self.value_function[next_s] * (not done)
                        )
                    else:
                        values.append(reward)
                self.value_function[s] = max(values)
                delta = max(delta, abs(v - self.value_function[s]))
            if delta < theta:
                break

        self._extract_policy(gamma)
        return self.policy

    def _get_next_state_reward(self, state, action):
        self.env.reset()
        self.env.player_pos, self.env.box_pos, self.env.storage_pos = state
        self.env.grid[self.env.player_pos[0], self.env.player_pos[1]] = self.env.PLAYER
        self.env.grid[self.env.box_pos[0], self.env.box_pos[1]] = self.env.BOX
        self.env.grid[self.env.storage_pos[0], self.env.storage_pos[1]] = (
            self.env.STORAGE
        )

        next_obs, reward, done, _, _ = self.env.step(action)
        next_state = self.env.get_state()

        return next_state, reward, done

    def _extract_policy(self, gamma):
        for s, state in enumerate(self.states):
            values = []
            for a in range(self.num_actions):
                next_state, reward, done = self._get_next_state_reward(state, a)
                if next_state in self.state_to_index:
                    next_s = self.state_to_index[next_state]
                    values.append(
                        reward + gamma * self.value_function[next_s] * (not done)
                    )
                else:
                    values.append(reward)
            self.policy[s] = np.argmax(values)

    def get_action(self, state):
        if state in self.state_to_index:
            return self.policy[self.state_to_index[state]]
        return np.random.randint(self.num_actions)
