import matplotlib.pyplot as plt
import numpy as np



class SokobanaValue:
    '''
    Class to solve Sokoban using Value Iteration.
    '''
    def __init__(self, grid, target_pos, actions, initial_states) -> None:
        self.grid = grid
        self.target_pos = target_pos
        self.actions = actions
        self.initial_states = initial_states
        self.target_reward = 0
        self.step_reward = -1



    def plot_grid(self) -> None:
        '''
        Function to plot the grid.
        '''
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'yellow', 'orange'])
        bounds = [0, 1, 2, 3, 4, 5, 6, 7]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(self.grid, cmap=cmap, norm=norm)

        plt.xticks([])
        plt.yticks([])
        plt.grid(which='both', color='gray', linestyle='-', linewidth=2)
        plt.gca().set_xticks(np.arange(-0.5, len(self.grid[0]), 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, len(self.grid), 1), minor=True)

        plt.gca().add_patch(plt.Rectangle((self.target_pos[1] - 0.5, self.target_pos[0] - 0.5), 1, 1, color='red'))

        plt.gca().add_patch(plt.Rectangle((self.initial_states[0][1] - 0.5, self.initial_states[0][0] - 0.5), 1, 1, color='yellow'))
        plt.gca().add_patch(plt.Rectangle((self.initial_states[1][1] - 0.5, self.initial_states[1][0] - 0.5), 1, 1, color='orange'))

        plt.show()


    
    def check_wall(self, pos) -> bool:
        '''
        Function to check if the position is a wall.

        Args:
        pos: tuple, position to check

        Returns:
        bool, True if the position is not a wall, False otherwise.
        '''
        if self.grid[pos[0]][pos[1]] != 1:
            return True
        
        return False
        

    
    def reward(self, state) -> int:
        '''
        Function to return the reward for the state.

        Args:
        state: tuple, state of the environment

        Returns:
        int, reward for the state
        '''
        _, box_position = state
        if box_position[0] == self.target_pos[0] and box_position[1] == self.target_pos[1]:
            return self.target_reward
        else:
            return self.step_reward
        
    

    def move(self, state, action) -> tuple:
        '''
        Function to move the player in the environment.

        Args:
        state: tuple, state of the environment
        action: str, action to take

        Returns:
        tuple, new state after taking the action
        int, reward for the new state
        '''
        agent_pos, box_pos = state
        move = self.actions[action]
        next_agent_pos = (agent_pos[0] + move[0], agent_pos[1] + move[1])
        if self.grid[next_agent_pos[0]][next_agent_pos[1]] == 1:
            return state, self.reward(state)
        if next_agent_pos == box_pos:
            next_box_pos = (next_agent_pos[0] + move[0], next_agent_pos[1] + move[1])
            if self.check_wall(next_box_pos):
                new_box_position = next_box_pos
                new_state = (next_agent_pos, new_box_position)
                return new_state, self.reward(new_state)
            else:
                return state, self.reward(state)
        else:
            new_state = (next_agent_pos, box_pos)
            return new_state, self.reward(new_state)
        
    

    def check_terminal(self, state) -> bool:
        '''
        Function to check if the state is terminal.

        Args:
        state: tuple, state of the environment

        Returns:
        bool, True if the state is terminal, False otherwise.
        '''
        _, box_pos = state
        if box_pos == self.target_pos:
            return True
        else:
            return False
        
    

    def value_iteration(self, theta, gamma) -> tuple:
        '''
        Function to perform Value Iteration.

        Args:
        theta: float, threshold for stopping the iteration
        gamma: float, discount factor

        Returns:
        dict, optimal policy
        dict, value function
        '''
        theta = theta
        gamma = gamma
        policy, V = {}, {}

        player_state_space = [(i, j) for i in range(len(self.grid)) for j in range(len(self.grid[0])) if (self.grid[i][j] != 1 and self.grid[i][j] != 3)]
        box_state_space = [(i, j) for i in range(len(self.grid)) for j in range(len(self.grid[0])) if (self.grid[i][j] != 1 and self.grid[i][j] != 3)]

        for player_state in player_state_space:
            for box_state in box_state_space:
                if player_state != box_state:
                    V[(player_state, box_state)] = 0
        
        while True:
            delta = 0
            for state in V.keys():
                if self.check_terminal(state):
                    continue
                v = V[state]
                max_value = float('-inf')

                for action in self.actions.keys():
                    next_state, reward = self.move(state, action)
                    value = reward + gamma * V[next_state]
                    if value > max_value:
                        max_value = value
                V[state] = max_value
                delta = max(delta, abs(v - V[state]))

            if delta < theta:
                break

        for state in V.keys():
            max_value = float('-inf')
            best_action = None
            for action in self.actions.keys():
                next_state, _ = self.move(state, action)
                value = V[next_state]
                if value > max_value:
                    max_value = value
                    best_action = action
            policy[state] = best_action

        return policy, V
        
        
    

    def test_policy(self, policy) -> None:
        '''
        Function to test the policy.

        Args:
        policy: dict, policy to test

        Returns:
        None
        '''
        state = self.initial_states
        steps = 0
        print("------------------------------------")
        print("------------------------------------")
        print("Testing Policy")
        print(f"Initial State: Player {state[0]}, Box {state[1]}")

        while not self.check_terminal(state):
            action = policy.get(state)
            if not action:
                print("Ending simulation.")
                break
            next_state, _ = self.move(state, action)
            print(f"Step {steps + 1}: Player moved {action}")
            state = next_state
            steps += 1
            if self.check_terminal(state):
                print(f"Terminal State reached: Player {state[0]}, Box {state[1]}")
                break
    


def main():
    # 1=wall, 0=floor, 3=goal
    grid = [
        [1, 1, 1, 1, 3, 3],
        [1, 0, 0, 1, 3, 3],
        [1, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 3, 3]
    ]

    target_pos = (3, 1)

    actions = {
        'UP': (-1, 0),
        'DOWN': (1, 0),
        'LEFT': (0, -1),
        'RIGHT': (0, 1)
    }

    initial_states = ((1, 2), (4, 3))

    sokobana = SokobanaValue(grid, target_pos, actions, initial_states)
    policy, V = sokobana.value_iteration(theta=1e-3, gamma=0.9)

    print("Optimal Policy:")
    for state, action in policy.items():
        print(f"State {state}: Take action {action}")
    
    print("\nValue Function:")
    for state, value in V.items():
        print(f"State {state}: Value {value}")

    sokobana.test_policy(policy)


if __name__ == '__main__':
    main()