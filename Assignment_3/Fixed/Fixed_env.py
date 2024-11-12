import matplotlib.pyplot as plt
import numpy as np

class FixedMAPFEnvironment:
    def __init__(self, grid_dimension=10, wall_config=None):
        self.grid_size = grid_dimension
        self.walls = wall_config or [
            (5,0), (5,1), (5,2), (4,2),
            (0,5), (1,5), (2,5), (2,4),
            (4,9), (4,8), (4,7), (5,7),
            (7,4), (7,5), (8,5), (9,5)
        ]
        self.agent_colors = ['orange', 'purple', 'green', 'pink']
        self.goal_pos = [(5,8), (1,4), (4,1), (8,4)]
        self.agent_pos = [(1,1), (8,1), (1,8), (8,8)]

    def initialize_scenario(self):
        """
        Set initial agent positions to fixed predefined locations
        
        Returns:
            list: Initial agent positions
        """
        self.agent_pos = [(1,1), (8,1), (8,8), (1,8)]
        return self.agent_pos

    def execute_action(self, current_positions, actions):
        """
        Execute actions for multiple agents
        
        Args:
            current_positions (list): Current agent positions
            actions (list): Actions for each agent
        
        Returns:
            tuple: Next positions, rewards, completion status
        """
        next_positions = []
        rewards = []

        for idx, (pos, action) in enumerate(zip(current_positions, actions)):
            proposed_pos = self._calculate_next_position(pos, action)
            
            if self._is_movement_valid(proposed_pos, next_positions):
                next_positions.append(proposed_pos)
            else:
                next_positions.append(pos)

            reward = 0 if next_positions[-1] == self.goal_pos[idx] else -1
            rewards.append(reward)

        self.agent_pos = next_positions
        completion_status = all(pos == goal for pos, goal in zip(next_positions, self.goal_pos))
        
        return next_positions, rewards, completion_status

    def _calculate_next_position(self, current_pos, action):
        """Calculate next position based on action"""
        x, y = current_pos
        if action == 0:  # Left
            return (x, max(0, y-1))
        elif action == 1:  # Right
            return (x, min(self.grid_size-1, y+1))
        elif action == 2:  # Up
            return (max(0, x-1), y)
        elif action == 3:  # Down
            return (min(self.grid_size-1, x+1), y)
        return current_pos  # Stay

    def _is_movement_valid(self, position, current_next_positions):
        """
        Validate if a movement is permissible
        
        Args:
            position (tuple): Proposed position
            current_next_positions (list): Current proposed next positions
        
        Returns:
            bool: Validity of movement
        """
        return position not in self.walls and position not in current_next_positions

    def plot_map(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Set up gridlines and limits
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        ax.grid(True, linewidth=2, color='lightgray')

        # Set aspect of the plot to be equal
        ax.set_aspect('equal')
        ax.set_xlabel('X-axis', fontsize=12, color='darkgray')
        ax.set_ylabel('Y-axis', fontsize=12, color='darkgray')

        # Remove the axes
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False, colors='darkgray')

        # Plot walls
        for (x, y) in self.walls:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='dimgray'))

        # Plot agents
        for i, (x, y) in enumerate(self.agent_pos):
            color = self.agent_colors[i]
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, alpha=0.8))
            ax.text(x + 0.5, y + 0.5, str(i), color='white', ha='center', va='center', fontsize=12)

        # Plot goals
        for i, (x, y) in enumerate(self.goal_pos):
            color = self.agent_colors[i]
            ax.plot(x + 0.5, y + 0.5, marker='+', color=color, mew=3, ms=20)

        plt.tight_layout()
        plt.savefig('mapf_env_fixed.png')
        plt.close()

# Main execution block
if __name__ == "__main__":
    np.random.seed(42)
    env = FixedMAPFEnvironment()
    env.plot_map()
    print("MAPF Environment with Fixed Positions initialized and visualized.")

