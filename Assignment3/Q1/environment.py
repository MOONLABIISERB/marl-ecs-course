import numpy as np
import matplotlib.pyplot as plt
import random
import time

class DiscreteActionSpace:
    """Defines a discrete action space with a sample() method."""
    def __init__(self, n):
        self.n = n  # Number of possible actions (0 to n-1)

    def sample(self):
        """Randomly samples an action from the action space."""
        return random.randint(0, self.n - 1)


class MAPFEnvironment:
    def __init__(self, grid_size, obstacles, agents, goals):
        """
        Initialize the MAPF environment.
        - grid_size: tuple (rows, cols)
        - obstacles: list of obstacle coordinates [(x1, y1), (x2, y2), ...]
        - agents: dict {agent_id: (x, y)}
        - goals: dict {agent_id: (x, y)}
        """
        self.fig, self.ax = None, None
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.initial_agents = agents
        self.initial_goals = goals
        self.action_space = DiscreteActionSpace(5)  # Actions: 0 (stay), 1 (up), 2 (down), 3 (left), 4 (right)
        self.agent_ids = sorted(agents.keys())  # Ensure consistent ordering of agents
        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.
        Returns:
        - obs: tuple of agent positions ((x1, y1), (x2, y2), ...)
        """
        self.agent_positions = {aid: pos for aid, pos in self.initial_agents.items()}
        self.goals = self.initial_goals
        self.terminated = {aid: False for aid in self.agent_positions}
        return tuple(self.agent_positions[aid] for aid in self.agent_ids)

    def is_valid_move(self, position):
        """Check if a position is valid (not an obstacle or out of bounds)."""
        x, y = position
        if x < 0 or y < 0 or x >= self.grid_size[0] or y >= self.grid_size[1]:
            return False
        if position in self.obstacles:
            return False
        return True

    def step(self, actions):
        """
        Perform a step in the environment.
        - actions: tuple of actions for each agent (action1, action2, ...)
          Actions are represented as:
          0: Stay
          1: Move up
          2: Move down
          3: Move left
          4: Move right
        
        Returns:
        - obs_: tuple of agent positions ((x1, y1), (x2, y2), ...)
        - reward: tuple of rewards for each agent (r1, r2, ...)
        - terminated: tuple of termination flags for each agent (t1, t2, ...)
        - truncated: always False (not applicable for this environment)
        - _: additional information (empty in this case)
        """
        new_positions = {}
        for aid, action in zip(self.agent_ids, actions):
            x, y = self.agent_positions[aid]
            if action == 1:  # Move up
                new_pos = (x - 1, y)
            elif action == 2:  # Move down
                new_pos = (x + 1, y)
            elif action == 3:  # Move left
                new_pos = (x, y - 1)
            elif action == 4:  # Move right
                new_pos = (x, y + 1)
            elif action == 0:  # Stay
                new_pos = (x, y)
            else:
                raise ValueError(f"Invalid action: {action}")

            # Validate move
            if self.is_valid_move(new_pos):
                new_positions[aid] = new_pos
            else:
                new_positions[aid] = (x, y)  # Stay in place if move is invalid

        # Resolve collisions: No two agents can occupy the same cell
        occupied_positions = {}
        for aid, pos in new_positions.items():
            if pos in occupied_positions:
                # If there's a collision, all involved agents stay in place
                new_positions[aid] = self.agent_positions[aid]
                new_positions[occupied_positions[pos]] = self.agent_positions[
                    occupied_positions[pos]
                ]
            else:
                occupied_positions[pos] = aid

        # Update positions and compute rewards
        reward = []
        terminated_flags = []
        for aid, new_pos in new_positions.items():
            self.agent_positions[aid] = new_pos
            if new_pos == self.goals[aid]:
                reward.append(0)  # No penalty for reaching the goal
                self.terminated[aid] = True
            else:
                reward.append(-1)  # Penalty for each step
            terminated_flags.append(self.terminated[aid])

        # Observation (current positions of agents)
        obs_ = tuple(self.agent_positions[aid] for aid in self.agent_ids)
        reward = tuple(reward)
        terminated = tuple(terminated_flags)
        truncated = False  # No truncation logic in this environment

        return obs_, reward, terminated, truncated, {}
    
    # def render(self):
    #     """Render the grid environment continuously using Matplotlib."""
    #     if self.fig is None or self.ax is None:
    #         # Initialize the figure and axes only once
    #         self.fig, self.ax = plt.subplots(figsize=(8, 8))
    #         plt.ion()  # Turn on interactive mode to allow continuous rendering

    #     self.ax.clear()  # Clear the axes for the new frame

    #     # Draw obstacles
    #     for i in range(len(self.obstacles)):
    #         x = self.obstacles[i][0]
    #         y = self.obstacles[i][1]
    #         self.ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color="gray"))

    #     # Draw goals (indicate with '+')
    #     for aid, (gx, gy) in self.goals.items():
    #         self.ax.text(gy, gx, "+", color=f"C{aid}", fontsize=20, 
    #                     ha="center", va="center", fontweight="bold")

    #     # Draw agents
    #     for aid, (x, y) in self.agent_positions.items():
    #         self.ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color=f"C{aid}", alpha=0.7))
    #         self.ax.text(y, x, str(aid), color="white", fontsize=12, 
    #                     ha="center", va="center")

    #     # Grid styling
    #     self.ax.set_xticks(np.arange(0, self.grid_size[1], 1))
    #     self.ax.set_yticks(np.arange(0, self.grid_size[0], 1))
    #     self.ax.set_xticks(np.arange(-0.5, self.grid_size[1], 1), minor=True)
    #     self.ax.set_yticks(np.arange(-0.5, self.grid_size[0], 1), minor=True)
    #     self.ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    #     self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    #     self.ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
    #     self.ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
    #     self.ax.invert_yaxis()  # Invert y-axis to match grid indexing

    #     # Update the plot with the current state
    #     self.fig.canvas.draw()      # Draw the updated figure
    #     self.fig.canvas.flush_events()  # Flush the GUI events
    #     plt.pause(0.05)  # Pause briefly to allow the update to be visible

    def render(self):
        """Render the grid environment in a separate window using Matplotlib."""
        fig, ax = plt.subplots(figsize=(9, 9))
        grid = np.zeros(self.grid_size, dtype=int)

        # Draw obstacles
        for x, y in self.obstacles:
            grid[x, y] = -1  # Obstacles
            ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color="gray"))

        # Draw goals (indicate with '+')
        for aid, (gx, gy) in self.goals.items():
            ax.text(gy, gx, "+", color=f"C{aid}", fontsize=20,
                    ha="center", va="center", fontweight="bold")

        # Draw agents
        for aid, (x, y) in self.agent_positions.items():
            grid[x, y] = aid  # Agents' IDs
            ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color=f"C{aid}", alpha=0.7))
            ax.text(y, x, str(aid), color="white", fontsize=12,
                    ha="center", va="center")

        # Grid styling
        ax.set_xticks(np.arange(0, self.grid_size[1], 1))
        ax.set_yticks(np.arange(0, self.grid_size[0], 1))
        ax.set_xticks(np.arange(-0.5, self.grid_size[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[0] - 0.5)

        # Show the plot in a separate window
        plt.gca().invert_yaxis()
        plt.show(block=True)  # Keeps the window open until manually closed


    
    # def render(self):
    #     """Render the grid environment using Matplotlib."""
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     grid = np.zeros(self.grid_size, dtype=int)
        
    #     print("Obstacles:", self.obstacles[0])
    #     # Draw obstacles
    #     for i in range(len(self.obstacles)):
    #         x = self.obstacles[i][0]
    #         y = self.obstacles[i][1]
    #         grid[x][y] = -1
            
    #     # # for x, y in self.obstacles[0]:
    #     # #     grid[x, y] = -1  # Obstacles

    #     # Draw goals (indicate with '+')
    #     for aid, (gx, gy) in self.goals.items():
    #         ax.text(gy, gx, "+", color=f"C{aid}", fontsize=20, 
    #                 ha="center", va="center", fontweight="bold")

    #     # Draw agents
    #     for aid, (x, y) in self.agent_positions.items():
    #         grid[x, y] = aid  # Agents' IDs
    #         ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, color=f"C{aid}", alpha=0.7))
    #         ax.text(y, x, str(aid), color="white", fontsize=12, 
    #                 ha="center", va="center")

    #     # Draw the grid with walls and colors
    #     for x in range(self.grid_size[0]):
    #         for y in range(self.grid_size[1]):
    #             if grid[x, y] == -1:  # Obstacles
    #                 ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, color="gray"))

    #     # Grid styling
    #     ax.set_xticks(np.arange(0, self.grid_size[1], 1))
    #     ax.set_yticks(np.arange(0, self.grid_size[0], 1))
    #     ax.set_xticks(np.arange(-0.5, self.grid_size[1], 1), minor=True)
    #     ax.set_yticks(np.arange(-0.5, self.grid_size[0], 1), minor=True)
    #     ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    #     ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    #     ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
    #     ax.set_ylim(-0.5, self.grid_size[0] - 0.5)

    #     # Show the plot
    #     plt.gca().invert_yaxis()
    #     plt.show()



