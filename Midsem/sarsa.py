import numpy as np
from modified_tsp import ModTSP

class SARSA:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.Q = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore
        else:
            return np.argmax(self.Q[state, :])  # Exploit

    def update_Q(self, state, action, reward, next_state, next_action):
        target = reward + self.gamma * self.Q[next_state, next_action]
        error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * error

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            action = self.choose_action(state)
            done = False

            while not done:
                next_state, reward, done, trun, _ = env.step(action)
                next_action = self.choose_action(next_state)
                self.Q = self.update_Q(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
        Q = self.Q
        return Q

test = SARSA(10, 10)
Q = test.train
print(Q)