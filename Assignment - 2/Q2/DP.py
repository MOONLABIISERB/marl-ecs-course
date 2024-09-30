import matplotlib.pyplot as plt
import numpy as np


class SokobanSolver:
    """
    Class to solve Sokoban using the Value Iteration method.
    """
    def __init__(self, grid_layout, goal_position, available_actions, start_states) -> None:
        self.grid_layout = grid_layout
        self.goal_position = goal_position
        self.available_actions = available_actions
        self.start_states = start_states
        self.reward_at_goal = 0
        self.penalty_per_step = -1


    def display_grid(self) -> None:
        """
        Function to visualize the grid.
        """
        colormap = plt.cm.colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'yellow', 'orange'])
        color_bounds = [0, 1, 2, 3, 4, 5, 6, 7]
        norm = plt.cm.colors.BoundaryNorm(color_bounds, colormap.N)

        plt.imshow(self.grid_layout, cmap=colormap, norm=norm)

        plt.xticks([])
        plt.yticks([])
        plt.grid(which='both', color='gray', linestyle='-', linewidth=2)
        plt.gca().set_xticks(np.arange(-0.5, len(self.grid_layout[0]), 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, len(self.grid_layout), 1), minor=True)

        plt.gca().add_patch(plt.Rectangle((self.goal_position[1] - 0.5, self.goal_position[0] - 0.5), 1, 1, color='red'))
        plt.gca().add_patch(plt.Rectangle((self.start_states[0][1] - 0.5, self.start_states[0][0] - 0.5), 1, 1, color='yellow'))
        plt.gca().add_patch(plt.Rectangle((self.start_states[1][1] - 0.5, self.start_states[1][0] - 0.5), 1, 1, color='orange'))

        plt.show()

    
    def is_valid_move(self, position) -> bool:
        """
        Function to verify if a position is not blocked.

        Args:
        position: tuple, the coordinates to check

        Returns:
        bool, True if it's a valid move, False if it's a wall.
        """
        return self.grid_layout[position[0]][position[1]] != 1

    
    def compute_reward(self, state) -> int:
        """
        Function to calculate the reward for a given state.

        Args:
        state: tuple, current environment state

        Returns:
        int, reward corresponding to the state
        """
        _, box_pos = state
        if box_pos == self.goal_position:
            return self.reward_at_goal
        return self.penalty_per_step
    

    def make_move(self, state, action) -> tuple:
        """
        Function to execute the player’s move.

        Args:
        state: tuple, current state of the environment
        action: str, action to execute

        Returns:
        tuple, new state after the move
        int, reward for the new state
        """
        player_pos, box_pos = state
        direction = self.available_actions[action]
        next_player_pos = (player_pos[0] + direction[0], player_pos[1] + direction[1])

        if self.grid_layout[next_player_pos[0]][next_player_pos[1]] == 1:
            return state, self.compute_reward(state)
        
        if next_player_pos == box_pos:
            next_box_pos = (next_player_pos[0] + direction[0], next_player_pos[1] + direction[1])
            if self.is_valid_move(next_box_pos):
                return (next_player_pos, next_box_pos), self.compute_reward((next_player_pos, next_box_pos))
            return state, self.compute_reward(state)
        return (next_player_pos, box_pos), self.compute_reward((next_player_pos, box_pos))
    

    def is_terminal(self, state) -> bool:
        """
        Function to determine if the current state is terminal.

        Args:
        state: tuple, current environment state

        Returns:
        bool, True if the state is terminal, otherwise False
        """
        _, box_pos = state
        return box_pos == self.goal_position
    

    def run_value_iteration(self, tolerance, discount) -> tuple:
        """
        Function to execute the Value Iteration algorithm.

        Args:
        tolerance: float, threshold for convergence
        discount: float, discount factor

        Returns:
        dict, optimal policy
        dict, computed value function
        """
        policy, value_function = {}, {}
        
        agent_states = [(i, j) for i in range(len(self.grid_layout)) for j in range(len(self.grid_layout[0])) if self.grid_layout[i][j] not in [1, 3]]
        box_states = [(i, j) for i in range(len(self.grid_layout)) for j in range(len(self.grid_layout[0])) if self.grid_layout[i][j] not in [1, 3]]

        for agent_pos in agent_states:
            for box_pos in box_states:
                if agent_pos != box_pos:
                    value_function[(agent_pos, box_pos)] = 0

        while True:
            delta = 0
            for state in value_function.keys():
                if self.is_terminal(state):
                    continue
                v = value_function[state]
                max_value = float('-inf')

                for action in self.available_actions.keys():
                    next_state, reward = self.make_move(state, action)
                    value = reward + discount * value_function[next_state]
                    if value > max_value:
                        max_value = value
                value_function[state] = max_value
                delta = max(delta, abs(v - value_function[state]))

            if delta < tolerance:
                break

        for state in value_function.keys():
            best_action, max_value = None, float('-inf')
            for action in self.available_actions.keys():
                next_state, _ = self.make_move(state, action)
                if value_function[next_state] > max_value:
                    max_value = value_function[next_state]
                    best_action = action
            policy[state] = best_action

        return policy, value_function
    

    def evaluate_policy(self, policy) -> None:
        """
        Function to test and display the outcome of a given policy.

        Args:
        policy: dict, policy to evaluate

        Returns:
        None
        """
        state = self.start_states
        step_count = 0
        print("------------------------------------")
        print("Evaluating Policy:")
        print(f"Start State: Player at {state[0]}, Box at {state[1]}")

        while not self.is_terminal(state):
            action = policy.get(state)
            if not action:
                print("No valid action, terminating test.")
                break
            next_state, _ = self.make_move(state, action)
            print(f"Step {step_count + 1}: Player moves {action}")
            state = next_state
            step_count += 1
            if self.is_terminal(state):
                print(f"Goal reached: Player at {state[0]}, Box at {state[1]}")
                break


def main():
    # 1=wall, 0=floor, 3=goal
    grid_layout = [
        [1, 1, 1, 1, 3, 3],
        [1, 0, 0, 1, 3, 3],
        [1, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 3, 3]
    ]

    goal_position = (3, 1)

    available_actions = {
        'UP': (-1, 0),
        'DOWN': (1, 0),
        'LEFT': (0, -1),
        'RIGHT': (0, 1)
    }

    start_states = ((1, 2), (4, 3))

    sokoban_solver = SokobanSolver(grid_layout, goal_position, available_actions, start_states)
    policy, value_function = sokoban_solver.run_value_iteration(tolerance=1e-3, discount=0.9)

    print("Optimal Policy:")
    for state, action in policy.items():
        print(f"State {state}: Perform action {action}")
    
    print("\nValue Function:")
    for state, value in value_function.items():
        print(f"State {state}: Value {value}")

    sokoban_solver.evaluate_policy(policy)


if __name__ == '__main__':
    main()
