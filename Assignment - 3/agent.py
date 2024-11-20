import numpy as np

class QLearningAgent:
    def __init__(self, agent_id, action_space, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.9):
        """
        Initialize the Q-learning agent.

        :param agent_id: Unique identifier for the agent.
        :param action_space: List of possible actions.
        :param exploration_rate: Probability of taking a random action (epsilon).
        :param learning_rate: Step size for updating Q-values (alpha).
        :param discount_factor: Weight for future rewards (gamma).
        """
        self.agent_id = agent_id
        self.action_space = action_space
        self.epsilon = exploration_rate
        self.alpha = learning_rate
        self.gamma = discount_factor

    def select_action(self, state, q_table):
        """
        Select an action using the epsilon-greedy strategy.

        :param state: Current state of the environment.
        :param q_table: Dictionary storing Q-values for state-action pairs.
        :return: Chosen action and the updated Q-table.
        """
        state_key = self._state_to_key(state)
        if state_key not in q_table:
            q_table[state_key] = {action: -float('inf') for action in self.action_space}

        if np.random.random() < self.epsilon:
            # Exploration: select a random action
            chosen_action = np.random.choice(self.action_space)
        else:
            # Exploitation: select the action with the highest Q-value
            chosen_action = max(q_table[state_key], key=q_table[state_key].get)

        return chosen_action, q_table

    def update_q_values(self, state, action, reward, next_state, done, q_table):
        """
        Update the Q-value for a given state-action pair using the Bellman equation.

        :param state: Current state of the environment.
        :param action: Action taken in the current state.
        :param reward: Reward received after taking the action.
        :param next_state: State transitioned to after taking the action.
        :param done: Whether the episode has ended.
        :param q_table: Dictionary storing Q-values for state-action pairs.
        :return: Updated Q-table.
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        if state_key not in q_table:
            q_table[state_key] = {action: -float('inf') for action in self.action_space}

        if next_state_key not in q_table:
            q_table[next_state_key] = {action: -float('inf') for action in self.action_space}

        current_q_value = q_table[state_key][action]
        max_future_q_value = max(q_table[next_state_key].values())

        # Calculate the target value
        if done:
            target = reward
        else:
            target = reward + self.gamma * max_future_q_value

        # Update Q-value using the learning rate
        q_table[state_key][action] = current_q_value + self.alpha * (target - current_q_value)

        return q_table

    def _state_to_key(self, state):
        """
        Convert the state into a hashable representation for the Q-table.

        :param state: Current state of the environment.
        :return: A tuple representing the state.
        """
        return tuple(state[self.agent_id])
