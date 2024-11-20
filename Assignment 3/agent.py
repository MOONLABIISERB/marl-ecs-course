import numpy as np

class QLearningAgent:
    def __init__(self, agent_id, action_space, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.agent_id = agent_id
        
        self.action_space = action_space
        self.epsilon = epsilon  # Exploration factor
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

    def choose_action(self, state, q_table):
        """ Epsilon-greedy action selection """
        state_key = self.get_state_key(state)
        if state_key not in q_table:
            q_table[state_key] = {a: -1000 for a in self.action_space}  # Initialize Q-values if state is new
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space), q_table # Explore: choose a random action
        else:
            q_values = q_table[state_key]
            return max(q_values, key=q_values.get), q_table # Action with highest Q-value

    def get_state_key(self, state):
        return tuple(state[self.agent_id])

    def learn(self, state, action, reward, next_state, done, q_table):
        """ Update Q-value based on the Bellman equation """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in q_table:
            q_table[state_key] = {a: -1000 for a in self.action_space}  # Initialize Q-values if state is new

        if next_state_key not in q_table:
            q_table[next_state_key] = {a: -1000 for a in self.action_space}  # Initialize Q-values if next state is new

        current_q = q_table[state_key][action]
        next_max_q = max(q_table[next_state_key].values())  # Max Q-value for next state
        target = reward + self.gamma * next_max_q

        # Update Q-value using the Bellman equation
        q_table[state_key][action] = current_q + self.alpha * (target - current_q)

        return q_table

