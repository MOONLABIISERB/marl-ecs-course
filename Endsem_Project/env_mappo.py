import gym
import numpy as np
from gym import spaces


class DenseTetheredBoatsEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        grid_size=10,
        n_boats=2,
        tether_length=3,
        time_penalty=-0.1,
        proximity_reward=0.5,
        trash_reward=10,
        complete_reward=50,
        step_per_episode=100,
    ):
        super(DenseTetheredBoatsEnv, self).__init__()

        # Environment parameters
        self.grid_size = grid_size
        self.n_boats = n_boats
        self.tether_length = tether_length

        # Rewards
        self.time_penalty = time_penalty
        self.proximity_reward = proximity_reward
        self.trash_reward = trash_reward
        self.complete_reward = complete_reward
        self.step_per_episode = step_per_episode

        # Grid values
        self.EMPTY = 0
        self.TRASH = 1
        self.BOAT = 2
        self.TETHER = 3

        # Action space: 9 actions per boat (8 directions + stay)
        self.action_space = spaces.MultiDiscrete([9] * n_boats)

        # Observation space: grid_size x grid_size with values 0-3
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0, high=3, shape=(grid_size, grid_size), dtype=np.int32
                ),
                "pos1": spaces.Box(
                    low=0, high=grid_size - 1, shape=(2,), dtype=np.int32
                ),
                "pos2": spaces.Box(
                    low=0, high=grid_size - 1, shape=(2,), dtype=np.int32
                ),
                "valid_actions": spaces.Tuple(
                    (
                        spaces.Box(low=0, high=1, shape=(9,), dtype=np.int32),
                        spaces.Box(low=0, high=1, shape=(9,), dtype=np.int32),
                    )
                ),
            }
        )

        self.reset()

    def _get_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))

    def _get_new_position(self, pos, action):
        """Get new position based on action"""
        x, y = pos
        # Action mapping: 0-7 for movement, 8 for stay
        moves = [
            (x + 1, y),  # 0: forward
            (x + 1, y - 1),  # 1: forward-left
            (x, y - 1),  # 2: left
            (x - 1, y - 1),  # 3: back-left
            (x - 1, y),  # 4: back
            (x - 1, y + 1),  # 5: back-right
            (x, y + 1),  # 6: right
            (x + 1, y + 1),  # 7: forward-right
            (x, y),  # 8: stay
        ]
        return moves[action]

    def _is_valid_move(self, current_pos, new_pos, other_boat_pos):
        """Check if move is valid"""
        x, y = new_pos

        # Check grid boundaries
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False

        # Check if boats would collide
        if new_pos == other_boat_pos:
            return False

        # Check tether length
        if self._get_distance(new_pos, other_boat_pos) > self.tether_length:
            return False

        return True

    def _get_tether_cells(self, boat1_pos, boat2_pos):
        """Calculate cells occupied by tether between boats"""
        x1, y1 = boat1_pos
        x2, y2 = boat2_pos

        points = []
        n_points = self.tether_length + 1

        for i in range(n_points):
            t = i / (n_points - 1)
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            points.append((x, y))

        return points

    def _get_valid_actions(self, agent_id=None):
        """Return valid action masks for both agents or specific agent"""
        if agent_id is not None:
            other_id = 1 - agent_id
            valid_actions = np.zeros(9, dtype=np.int32)

            for action in range(9):
                new_pos = self._get_new_position(self.boat_positions[agent_id], action)
                if self._is_valid_move(
                    self.boat_positions[agent_id],
                    new_pos,
                    self.boat_positions[other_id],
                ):
                    valid_actions[action] = 1

            return valid_actions
        else:
            return (self._get_valid_actions(0), self._get_valid_actions(1))

    def _get_min_trash_distance(self, pos):
        """Get distance to nearest trash piece"""
        if not self.trash_positions:
            return float("inf")
        distances = [
            self._get_distance(pos, trash_pos) for trash_pos in self.trash_positions
        ]
        return min(distances)

    def step_agent(self, agent_id, action):
        """Execute action for single agent and return intermediate state"""
        # Save old position for reward calculation
        old_pos = self.boat_positions[agent_id]

        # Execute move
        new_pos = self._get_new_position(old_pos, action)
        other_id = 1 - agent_id

        if self._is_valid_move(old_pos, new_pos, self.boat_positions[other_id]):
            self.boat_positions[agent_id] = new_pos

        # Update grid state
        self.update_grid_state()

        # Return intermediate state
        return self.get_state_dict()

    def update_grid_state(self):
        """Update the grid state based on current positions"""
        self.grid.fill(self.EMPTY)

        # Place trash
        for trash_pos in self.trash_positions:
            self.grid[trash_pos] = self.TRASH

        # Place boats
        for boat_pos in self.boat_positions:
            self.grid[boat_pos] = self.BOAT

        # Place tether
        for tether_pos in self._get_tether_cells(*self.boat_positions):
            if self.grid[tether_pos] != self.BOAT:  # Don't overwrite boats
                self.grid[tether_pos] = self.TETHER

    def get_state_dict(self):
        """Return the current state as a dictionary"""
        return {
            "grid": self.grid.copy(),
            "pos1": np.array(self.boat_positions[0]),
            "pos2": np.array(self.boat_positions[1]),
            "valid_actions": self._get_valid_actions(),
        }

    def step(self, action):
        """Execute one time step"""
        assert self.action_space.contains(action)

        self.step_num += 1
        reward = self.time_penalty  # Base time penalty

        # Store old positions and distances
        old_positions = self.boat_positions.copy()
        old_distances = [self._get_min_trash_distance(pos) for pos in old_positions]

        # Execute actions sequentially
        self.step_agent(0, action[0])
        self.step_agent(1, action[1])

        # Calculate proximity rewards
        new_distances = [
            self._get_min_trash_distance(pos) for pos in self.boat_positions
        ]
        for old_dist, new_dist in zip(old_distances, new_distances):
            if new_dist < old_dist:  # If got closer to trash
                reward += self.proximity_reward * (old_dist - new_dist)

        # Check for trash collection
        trash_collected = []
        tether_cells = self._get_tether_cells(*self.boat_positions)

        for trash_pos in self.trash_positions:
            # Check if boats collect trash
            if trash_pos in self.boat_positions:
                trash_collected.append(trash_pos)
                reward += self.trash_reward
            # Check if tether collects trash
            elif trash_pos in tether_cells:
                trash_collected.append(trash_pos)
                reward += self.trash_reward

        # Remove collected trash
        self.trash_positions = [
            pos for pos in self.trash_positions if pos not in trash_collected
        ]

        # Update grid state
        self.update_grid_state()

        # Check if done
        done = len(self.trash_positions) == 0 or self.step_num >= self.step_per_episode

        # Add completion reward if all trash collected
        if len(self.trash_positions) == 0:
            reward += self.complete_reward

        return self.get_state_dict(), reward, done, {}

    def reset(self):
        """Reset environment state"""
        self.step_num = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Initialize boat positions
        self.boat_positions = [(0, 0), (0, self.tether_length)]

        # Initialize random trash positions
        n_trash = self.grid_size
        self.trash_positions = []
        while len(self.trash_positions) < n_trash:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(self.grid_size // 2, self.grid_size)
            pos = (x, y)
            if pos not in self.trash_positions and pos not in self.boat_positions:
                self.trash_positions.append(pos)

        # Update grid state
        self.update_grid_state()

        return self.get_state_dict()

    def render(self, mode="human"):
        """Simple console rendering"""
        if mode == "human":
            symbols = {
                self.EMPTY: ".",
                self.TRASH: "T",
                self.BOAT: "B",
                self.TETHER: "*",
            }

            for row in self.grid:
                print(" ".join(symbols[cell] for cell in row))
            print(
                f"Step: {self.step_num}, Trash remaining: {len(self.trash_positions)}"
            )
            print()


if __name__ == "__main__":
    # Test environment
    env = DenseTetheredBoatsEnv()
    obs = env.reset()
    env.render()

    # Test a few random steps
    for _ in range(5):
        valid_actions = obs["valid_actions"]
        action = [
            np.random.choice(np.where(valid_actions[0])[0]),
            np.random.choice(np.where(valid_actions[1])[0]),
        ]
        obs, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}")
        env.render()
        if done:
            break
