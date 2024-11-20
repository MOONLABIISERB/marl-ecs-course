import numpy as np
import random
class Q_learning:
    def __init__(self, grid_size, n_actions):
        self.grid_size = grid_size
        self.n_actions = n_actions

    def create_q(self):
        states = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        q_table = {}  # Using a dictionary to store Q-values
        for state in states:
            for action in range(self.n_actions):
                q_table[(state, action)] = 0.0  # Initialize Q-value for each state-action pair
        return q_table
    
    def get_max(self, Q, state):
        max_value = -float('inf')
        max_action = None
        for b in range(self.n_actions):
            max_value = max(max_value, Q[(state, b)])
        return max_value
    
    def best_action(self, Q, state):
        max_value = -float('inf')
        best_action = None
        for b in range(self.n_actions):
            if (state, b) in Q:
                current_value = Q[(state, b)]
                if current_value > max_value:
                    max_value = current_value
                    best_action = b
        return best_action            
        

    def epsilon_greedy(self, state, epsilon, Q):
        a = random.random()
        if a < epsilon:
            action = np.random.randint(0, self.n_actions-1)
        else:
            action = self.best_action(Q, state)    
        return action    


   