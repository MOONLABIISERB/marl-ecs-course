import numpy as np

class TSP_DP:
    def __init__(self, num_targets: int, distances: np.ndarray, discount_factor=0.99, threshold=1e-4):
        self.num_targets = num_targets
        self.distances = distances
        self.V = np.zeros(num_targets)  # State-value function
        self.policy = np.zeros(num_targets, dtype=int)  # Policy: best action per state
        self.discount_factor = discount_factor
        self.threshold = threshold

    def value_iteration(self):
        """Performs value iteration to find optimal values and policy."""
        while True:
            delta = 0
            for s in range(self.num_targets):
                v = self.V[s]
                action_values = self._one_step_lookahead(s)
                self.V[s] = np.min(action_values)  # Minimize distance (cost)
                self.policy[s] = np.argmin(action_values)  # Get the optimal action
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.threshold:
                break
        return self.V, self.policy

    def _one_step_lookahead(self, state):
        """Helper function to calculate expected values for each action."""
        action_values = np.zeros(self.num_targets)
        for action in range(self.num_targets):
            reward = -self.distances[state][action]  # Negative distance as reward
            action_values[action] = reward + self.discount_factor * self.V[action]
        return action_values

# Example usage:
num_targets = 5
locations = np.random.rand(num_targets, 2) * 10  # Random points in a 10x10 grid
distances = np.zeros((num_targets, num_targets))
for i in range(num_targets):
    for j in range(num_targets):
        distances[i, j] = np.linalg.norm(locations[i] - locations[j])

tsp_dp = TSP_DP(num_targets, distances)
optimal_values, optimal_policy = tsp_dp.value_iteration()

print("Optimal Values (DP):", optimal_values)
print("Optimal Policy (DP):", optimal_policy)
