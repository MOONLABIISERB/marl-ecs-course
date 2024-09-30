import numpy as np

class SokobanSolver:
    '''
    A class to tackle the Sokoban puzzle using Monte Carlo methods.
    '''
    def __init__(self, level_layout):
        self.level_layout = level_layout
        self.grid_height = len(level_layout)
        self.grid_width = len(level_layout[0])
        self.moves = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        self.initialize_environment()

    def initialize_environment(self):
        '''
        Reset the environment to its starting configuration.

        Returns:
        tuple, the initial state
        '''
        self.walls = []
        self.targets = []
        self.starting_player_pos = None
        self.starting_box_positions = []

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                current_cell = self.level_layout[row][col]
                if current_cell == '#':
                    self.walls.append((row, col))
                elif current_cell == '.':
                    self.targets.append((row, col))
                elif current_cell == '@' or current_cell == '+':
                    self.starting_player_pos = (row, col)
                    if current_cell == '+':
                        self.targets.append((row, col))
                elif current_cell == '$' or current_cell == '*':
                    self.starting_box_positions.append((row, col))
                    if current_cell == '*':
                        self.targets.append((row, col))
        
        self.player_pos = self.starting_player_pos
        self.box_positions = self.starting_box_positions.copy()
        self.step_count = 0
        return self.get_current_state()

    def get_current_state(self):
        '''
        Retrieve the current game state as a tuple.

        Returns:
        tuple, representing the current state
        '''
        return (self.player_pos, tuple(sorted(self.box_positions)))

    def execute_move(self, direction):
        '''
        Move the player and return the new state, reward, and completion status.

        Args:
        direction: str, direction of movement

        Returns:
        tuple: new state
        int: reward for the action
        bool: True if all boxes are correctly placed, False otherwise
        '''
        move_delta = self.moves[direction]
        delta_row, delta_col = move_delta
        current_row, current_col = self.player_pos
        new_row, new_col = current_row + delta_row, current_col + delta_col
        new_pos = (new_row, new_col)

        if not (0 <= new_row < self.grid_height and 0 <= new_col < self.grid_width) or new_pos in self.walls:
            return self.get_current_state(), -1, self.is_completed()

        if new_pos in self.box_positions:
            new_box_row, new_box_col = new_row + delta_row, new_col + delta_col
            new_box_pos = (new_box_row, new_box_col)

            if (not (0 <= new_box_row < self.grid_height and 0 <= new_box_col < self.grid_width) or
                new_box_pos in self.walls or new_box_pos in self.box_positions):
                return self.get_current_state(), -1, self.is_completed()

            self.box_positions.remove(new_pos)
            self.box_positions.append(new_box_pos)

        self.player_pos = new_pos
        self.step_count += 1
        return self.get_current_state(), -1 if not self.is_completed() else 0, self.is_completed()

    def is_completed(self):
        '''
        Check if all boxes are on target positions.

        Returns:
        bool, True if all boxes are in place, otherwise False
        '''
        return sorted(self.box_positions) == sorted(self.targets)

    def simulate_episode(self, strategy, max_turns=1000):
        '''
        Run a simulation episode using the provided strategy.

        Args:
        strategy: function, policy to determine actions
        max_turns: int, limit on the number of moves

        Returns:
        list, a log of (state, action, reward) for the episode
        '''
        state = self.initialize_environment()
        episode_log = []
        for _ in range(max_turns):
            action = strategy(state)
            next_state, reward, completed = self.execute_move(action)
            episode_log.append((state, action, reward))
            if completed:
                break
            state = next_state
        return episode_log

    def monte_carlo(self, strategy, episodes=1000, max_turns=1000, gamma=0.9, first_visit=True):
        '''
        Run Monte Carlo simulations to evaluate states.

        Args:
        strategy: function, action selection policy
        episodes: int, number of simulation runs
        max_turns: int, limit on steps per episode
        gamma: float, discount factor
        first_visit: bool, if True, use First-Visit MC, else use Every-Visit MC

        Returns:
        dict, mapping from states to values
        '''
        total_returns = {}
        visit_count = {}
        state_values = {}

        for _ in range(episodes):
            episode = self.simulate_episode(strategy, max_turns)
            visited_states = set()
            G = 0
            for t in reversed(range(len(episode))):
                state, _, reward = episode[t]
                G = gamma * G + reward
                if first_visit and state in visited_states:
                    continue
                visited_states.add(state)
                if state not in total_returns:
                    total_returns[state] = 0.0
                    visit_count[state] = 0
                total_returns[state] += G
                visit_count[state] += 1
                state_values[state] = total_returns[state] / visit_count[state]
        return state_values

    def optimize_policy(self, state_values, gamma=0.9):
        '''
        Improve the policy based on the current state values.

        Args:
        state_values: dict, state-value function
        gamma: float, discount factor

        Returns:
        dict, optimized policy mapping states to best actions
        '''
        new_policy = {}
        for state in state_values.keys():
            best_value = float('-inf')
            optimal_move = None
            self.player_pos, self.box_positions = state[0], list(state[1])
            for action in self.moves.keys():
                saved_pos = self.player_pos
                saved_boxes = self.box_positions.copy()
                next_state, reward, _ = self.execute_move(action)
                total_value = reward + gamma * state_values.get(next_state, 0)
                self.player_pos = saved_pos
                self.box_positions = saved_boxes
                if total_value > best_value:
                    best_value = total_value
                    optimal_move = action
            new_policy[state] = optimal_move
        return new_policy

def random_strategy(state):
    ''' 
    Returns a random move.

    Args:
    state: tuple, the current game state

    Returns:
    str, a random action
    '''
    return np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])

level_design = [
    ['#', '#', '#', '#', '#', '#'],
    ['#', ' ', '@', '#', '#', '#'],
    ['#', ' ', ' ', '#', '#', '#'],
    ['#', '.', ' ', ' ', ' ', '#'],
    ['#', ' ', ' ', '$', ' ', '#'],
    ['#', ' ', ' ', '#', '#', '#'],
    ['#', '#', '#', '#', '#', '#'],
]

def main():
    environment = SokobanSolver(level_design)
    random_policy = random_strategy

    print("Running Monte Carlo (First-Visit)...")
    V_first = environment.monte_carlo(random_policy, episodes=1000)

    print("Improving strategy with First-Visit Monte Carlo values...")
    improved_policy_first = environment.optimize_policy(V_first)

    print("Running Monte Carlo (Every-Visit)...")
    V_every = environment.monte_carlo(random_policy, episodes=1000, first_visit=False)

    print("Improving strategy with Every-Visit Monte Carlo values...")
    improved_policy_every = environment.optimize_policy(V_every)

    print("\nOptimized Policy (First-Visit):")
    for state, action in improved_policy_first.items():
        print(f"State: {state}, Best Move: {action}")

    print("\nOptimized Policy (Every-Visit):")
    for state, action in improved_policy_every.items():
        print(f"State: {state}, Best Move: {action}")

if __name__ == "__main__":
    main()
