import numpy as np

class SokobanSolver:
    def __init__(self, level_layout):
        self.level_layout = level_layout
        self.grid_height, self.grid_width = len(level_layout), len(level_layout[0])
        self.moves = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        self.initialize_environment()

    def initialize_environment(self):
        self.walls, self.targets, self.starting_box_positions = [], [], []
        for row, line in enumerate(self.level_layout):
            for col, cell in enumerate(line):
                if cell == '#': self.walls.append((row, col))
                elif cell == '.': self.targets.append((row, col))
                elif cell in '@+': self.starting_player_pos = (row, col); 
                if cell in '+*': self.targets.append((row, col))
                if cell in '$*': self.starting_box_positions.append((row, col))
        self.player_pos, self.box_positions = self.starting_player_pos, self.starting_box_positions.copy()
        return self.get_current_state()

    def get_current_state(self):
        return self.player_pos, tuple(sorted(self.box_positions))

    def execute_move(self, direction):
        delta_row, delta_col = self.moves[direction]
        new_pos = (self.player_pos[0] + delta_row, self.player_pos[1] + delta_col)
        if new_pos in self.walls or not (0 <= new_pos[0] < self.grid_height and 0 <= new_pos[1] < self.grid_width):
            return self.get_current_state(), -1, self.is_completed()

        if new_pos in self.box_positions:
            new_box_pos = (new_pos[0] + delta_row, new_pos[1] + delta_col)
            if new_box_pos in self.walls or new_box_pos in self.box_positions:
                return self.get_current_state(), -1, self.is_completed()
            self.box_positions.remove(new_pos)
            self.box_positions.append(new_box_pos)

        self.player_pos = new_pos
        return self.get_current_state(), -1 if not self.is_completed() else 0, self.is_completed()

    def is_completed(self):
        return sorted(self.box_positions) == sorted(self.targets)

    def simulate_episode(self, strategy, max_turns=1000):
        state, episode_log = self.initialize_environment(), []
        for _ in range(max_turns):
            action = strategy(state)
            next_state, reward, completed = self.execute_move(action)
            episode_log.append((state, action, reward))
            if completed: break
            state = next_state
        return episode_log

    def monte_carlo(self, strategy, episodes=1000, max_turns=1000, gamma=0.9, first_visit=True):
        total_returns, visit_count, state_values = {}, {}, {}
        for _ in range(episodes):
            episode, visited_states, G = self.simulate_episode(strategy, max_turns), set(), 0
            for t in reversed(range(len(episode))):
                state, _, reward = episode[t]
                G = gamma * G + reward
                if first_visit and state in visited_states: continue
                visited_states.add(state)
                total_returns[state] = total_returns.get(state, 0) + G
                visit_count[state] = visit_count.get(state, 0) + 1
                state_values[state] = total_returns[state] / visit_count[state]
        return state_values

    def optimize_policy(self, state_values, gamma=0.9):
        new_policy = {}
        for state in state_values:
            best_value, optimal_move = float('-inf'), None
            self.player_pos, self.box_positions = state[0], list(state[1])
            for action in self.moves:
                saved_pos, saved_boxes = self.player_pos, self.box_positions.copy()
                next_state, reward, _ = self.execute_move(action)
                value = reward + gamma * state_values.get(next_state, 0)
                self.player_pos, self.box_positions = saved_pos, saved_boxes
                if value > best_value: best_value, optimal_move = value, action
            new_policy[state] = optimal_move
        return new_policy

def random_strategy(state):
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
    print("Running Monte Carlo (First-Visit)...")
    V_first = environment.monte_carlo(random_strategy, episodes=1000)
    print("Improving strategy with First-Visit Monte Carlo values...")
    improved_policy_first = environment.optimize_policy(V_first)

    print("Running Monte Carlo (Every-Visit)...")
    V_every = environment.monte_carlo(random_strategy, episodes=1000, first_visit=False)
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
