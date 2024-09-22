import numpy as np

class TSP_MC:
    def __init__(self, num_targets: int, distances: np.ndarray, discount_factor=0.99):
        self.num_targets = num_targets
        self.distances = distances
        self.discount_factor = discount_factor
        self.Q = np.zeros((num_targets, num_targets))  # Action-value table
        self.returns_sum = np.zeros((num_targets, num_targets))  # Sum of returns
        self.returns_count = np.zeros((num_targets, num_targets))  # Count of returns

    def monte_carlo(self, num_episodes=1000, exploring_starts=True, first_visit=True):
        """Monte Carlo with Exploring Starts, supports both first-visit and every-visit methods."""
        for episode in range(num_episodes):
            # Generate an episode
            episode_data = self._generate_episode(exploring_starts)
            G = 0  # Initialize the return
            visited_state_action_pairs = set()

            # Iterate backward through the episode
            for t in reversed(range(len(episode_data))):
                state, action, reward = episode_data[t]
                G = self.discount_factor * G + reward  # Update return

                # First-visit vs every-visit check
                if first_visit and (state, action) in visited_state_action_pairs:
                    continue
                visited_state_action_pairs.add((state, action))

                # Update Q-value and returns for the state-action pair
                self.returns_sum[state, action] += G
                self.returns_count[state, action] += 1
                self.Q[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]

        # Derive the policy from the Q-values (choose the action with the highest Q-value)
        policy = np.argmax(self.Q, axis=1)
        return self.Q, policy

    def _generate_episode(self, exploring_starts=True):
        """Generates an episode with random exploring starts."""
        episode = []
        state = np.random.choice(self.num_targets)  # Start from a random target (exploring starts)
        visited = set()

        # Loop until all targets are visited
        for _ in range(self.num_targets):
            if state in visited:
                continue
            action = np.random.choice(self.num_targets)  # Random action (next target)
            reward = -self.distances[state, action] if action not in visited else -10000  # Penalty for revisiting
            episode.append((state, action, reward))
            visited.add(action)  # Mark target as visited
            state = action  # Move to the next state (target)
        return episode

# Example usage:

# Initialize environment parameters
num_targets = 5  # Define number of targets
locations = np.random.rand(num_targets, 2) * 10  # Random points in a 10x10 grid
distances = np.zeros((num_targets, num_targets))

# Calculate pairwise distances between the targets
for i in range(num_targets):
    for j in range(num_targets):
        distances[i, j] = np.linalg.norm(locations[i] - locations[j])

# Initialize the Monte Carlo solver
tsp_mc = TSP_MC(num_targets, distances)

# Perform Monte Carlo with exploring starts and first-visit method
mc_q_values_first_visit, mc_policy_first_visit = tsp_mc.monte_carlo(first_visit=True)
print("Q-values (MC First-Visit):", mc_q_values_first_visit)
print("Optimal Policy (MC First-Visit):", mc_policy_first_visit)

# Perform Monte Carlo with exploring starts and every-visit method
mc_q_values_every_visit, mc_policy_every_visit = tsp_mc.monte_carlo(first_visit=False)
print("Q-values (MC Every-Visit):", mc_q_values_every_visit)
print("Optimal Policy (MC Every-Visit):", mc_policy_every_visit)
