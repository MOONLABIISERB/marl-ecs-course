import numpy as np

class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.internal_memory = {
            "states": [],
            "next_states": [],
            "actions": [],
            "rewards": [],
            "penalties": []  # Track penalties in memory
        }

    def __len__(self):
        return len(self.internal_memory["states"])

    def add(self, state, next_state, action, reward, penalty=0):
        """
        Add a new entry to the memory
        Args:
            state: The current state
            next_state: The state after taking the action
            action: The action taken
            reward: The reward received
            penalty: Additional penalty for colliding with obstacles (default is 0)
        """
        # If too large, remove the oldest entries to make room
        if len(self) == self.size:
            self.internal_memory["states"].pop(0)
            self.internal_memory["next_states"].pop(0)
            self.internal_memory["actions"].pop(0)
            self.internal_memory["rewards"].pop(0)
            self.internal_memory["penalties"].pop(0)  # Remove oldest penalty as well

        # Add the new entry (with the penalty)
        self.internal_memory["states"].append(state)
        self.internal_memory["next_states"].append(next_state)
        self.internal_memory["actions"].append(action)
        self.internal_memory["rewards"].append(reward)
        self.internal_memory["penalties"].append(penalty)  # Store penalty with other memory

    def get_batch(self, batch_size, shuffle=True):
        """
        Retrieve a batch of experiences from memory.
        Args:
            batch_size: The number of experiences to return
            shuffle: If True, shuffles the batch before sampling. Defaults to True.
        Returns: 
            state_batch, next_state_batch, action_batch, reward_batch, penalty_batch
        """
        if len(self) < 10 * batch_size:
            return None  # Not enough data to form a batch

        # Generate a random permutation of indices
        permutation = np.arange(0, len(self))
        if shuffle:
            np.random.shuffle(permutation)

        # Select a random batch of indices
        batch_mask = permutation[-batch_size:]

        # Prepare the batches for state, next state, action, reward, and penalty
        state_batch = np.array([self.internal_memory["states"][k] for k in batch_mask])
        next_state_batch = np.array([self.internal_memory["next_states"][k] for k in batch_mask])
        action_batch = np.array([self.internal_memory["actions"][k] for k in batch_mask])
        reward_batch = np.array([self.internal_memory["rewards"][k] for k in batch_mask])
        penalty_batch = np.array([self.internal_memory["penalties"][k] for k in batch_mask])

        return state_batch, next_state_batch, action_batch, reward_batch, penalty_batch

