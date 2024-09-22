import numpy as np

class SokobanEnvironment:
    '''
    Class to solve Sokoban using Monte Carlo.
    '''
    def __init__(self, level) -> None:
        self.level = level
        self.GRID_HEIGHT = len(level)
        self.GRID_WIDTH = len(level[0])
        self.actions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        self.reset()



    def reset(self) -> tuple:
        '''
        Function to reset the environment to the initial state.

        Returns:
        tuple, initial state
        '''
        self.walls = []
        self.goals = []
        self.initial_player_pos = None
        self.initial_box_positions = []
        for row in range(self.GRID_HEIGHT):
            for col in range(self.GRID_WIDTH):
                cell = self.level[row][col]
                if cell == '#':
                    self.walls.append((row, col))
                elif cell == '.':
                    self.goals.append((row, col))
                elif cell == '@':
                    self.initial_player_pos = (row, col)
                elif cell == '+':
                    self.initial_player_pos = (row, col)
                    self.goals.append((row, col))
                elif cell == '$':
                    self.initial_box_positions.append((row, col))
                elif cell == '*':
                    self.initial_box_positions.append((row, col))
                    self.goals.append((row, col))
        self.player_pos = self.initial_player_pos
        self.box_positions = self.initial_box_positions.copy()
        self.steps = 0
        return self.get_state()



    def get_state(self) -> tuple:
        '''
        Function to return the current state as a tuple.

        Returns:
        tuple, current state
        '''
        sorted_boxes = tuple(sorted(self.box_positions))
        return (self.player_pos, sorted_boxes)



    def step(self, action) -> tuple:
        '''
        Function to take an action and return the next state, reward, and done flag.

        Args:
        action: str, action to take

        Returns:
        tuple, next state
        int, reward for the next state
        bool, flag to indicate if the episode is done
        '''
        move = self.actions[action]
        row_delta, col_delta = move
        player_row, player_col = self.player_pos
        new_player_row = player_row + row_delta
        new_player_col = player_col + col_delta
        new_player_pos = (new_player_row, new_player_col)

        if not (0 <= new_player_row < self.GRID_HEIGHT and 0 <= new_player_col < self.GRID_WIDTH):
            return self.get_state(), -1, self.is_done()

        if new_player_pos in self.walls:
            return self.get_state(), -1, self.is_done()

        if new_player_pos in self.box_positions:
            box_new_row = new_player_row + row_delta
            box_new_col = new_player_col + col_delta
            box_new_pos = (box_new_row, box_new_col)

            if not (0 <= box_new_row < self.GRID_HEIGHT and 0 <= box_new_col < self.GRID_WIDTH):
                return self.get_state(), -1, self.is_done()

            if box_new_pos in self.walls or box_new_pos in self.box_positions:
                return self.get_state(), -1, self.is_done()

            self.box_positions.remove(new_player_pos)
            self.box_positions.append(box_new_pos)

            self.player_pos = new_player_pos
            self.steps += 1
            done = self.is_done()
            reward = 0 if done else -1
            return self.get_state(), reward, done
        else:
            self.player_pos = new_player_pos
            self.steps += 1
            done = self.is_done()
            reward = 0 if done else -1
            return self.get_state(), reward, done



    def is_done(self) -> bool:
        '''
        Function to check if all boxes are on goal positions.

        Returns:
        bool, True if all boxes are on goal positions, False otherwise.
        '''
        return sorted(self.box_positions) == sorted(self.goals)




    def run_episode(self, policy, max_steps=1000) -> list:
        '''
        Function to run an episode using the given policy.

        Args:
        policy: function, policy to use
        max_steps: int, maximum number of steps to run the episode

        Returns:
        list, episode containing (state, action, reward) tuples
        '''
        state = self.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done = self.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode



    def monte_carlo_first_visit(self, policy, gamma=0.9, episodes=1000, max_steps=1000) -> dict:
        '''
        Function to run the First-Visit Monte Carlo method.

        Args:
        policy: function, policy to use
        gamma: float, discount factor
        episodes: int, number of episodes to run
        max_steps: int, maximum number of steps to run the episode

        Returns:
        dict, value function V
        '''
        returns_sum = {}
        returns_count = {}
        V = {}
        for i in range(episodes):
            episode = self.run_episode(policy, max_steps)
            visited_states = set()
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = gamma * G + reward
                if state not in visited_states:
                    visited_states.add(state)
                    if state not in returns_sum:
                        returns_sum[state] = 0.0
                        returns_count[state] = 0
                    returns_sum[state] += G
                    returns_count[state] += 1
                    V[state] = returns_sum[state] / returns_count[state]
        return V



    def monte_carlo_every_visit(self, policy, gamma=0.9, episodes=1000, max_steps=1000) -> dict:
        '''
        Function to run the Every-Visit Monte Carlo method.

        Args:
        policy: function, policy to use
        gamma: float, discount factor
        episodes: int, number of episodes to run
        max_steps: int, maximum number of steps to run the episode

        Returns:
        dict, value function V
        '''
        returns_sum = {}
        returns_count = {}
        V = {}
        for i in range(episodes):
            episode = self.run_episode(policy, max_steps)
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = gamma * G + reward
                if state not in returns_sum:
                    returns_sum[state] = 0.0
                    returns_count[state] = 0
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]
        return V



    def improve_policy(self, V, gamma=0.9)-> dict:
        '''
        Function to improve the policy based on the value function.

        Args:
        V: dict, value function
        gamma: float, discount factor

        Returns:
        dict, improved policy
        '''
        policy = {}
        all_states = V.keys()
        for state in all_states:
            self.player_pos, self.box_positions = state[0], list(state[1])
            best_action = None
            best_value = float('-inf')
            for action in self.actions.keys():
                # Save current state to restore later
                saved_player_pos = self.player_pos
                saved_box_positions = self.box_positions.copy()
                next_state, reward, done = self.step(action)
                next_state_value = V.get(next_state, 0)
                total_value = reward + gamma * next_state_value

                # Restore the state
                self.player_pos = saved_player_pos
                self.box_positions = saved_box_positions

                if total_value > best_value:
                    best_value = total_value
                    best_action = action
            policy[state] = best_action
        return policy



def random_policy(state) -> str:
    '''
    Function to return a random action.

    Args:
    state: tuple, current state

    Returns:
    str, random action
    '''
    return np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])



level = [
    ['#', '#', '#', '#', '#', '#'],
    ['#', ' ', '@', '#', '#', '#'],
    ['#', ' ', ' ', '#', '#', '#'],
    ['#', '.', ' ', ' ', ' ', '#'],
    ['#', ' ', ' ', '$', ' ', '#'],
    ['#', ' ', ' ', '#', '#', '#'],
    ['#', '#', '#', '#', '#', '#'],
]

def main():
    env = SokobanEnvironment(level)
    policy = random_policy

    print("Running First-Visit Monte Carlo...")
    V_first_visit = env.monte_carlo_first_visit(policy, episodes=1000)

    print("\nImproving policy based on First-Visit Value Function...")
    improved_policy_first_visit = env.improve_policy(V_first_visit)

    print("\nRunning Every-Visit Monte Carlo...")
    V_every_visit = env.monte_carlo_every_visit(policy, episodes=1000)

    print("\nImproving policy based on Every-Visit Value Function...")
    improved_policy_every_visit = env.improve_policy(V_every_visit)

    print("\nImproved Policy from First-Visit Monte Carlo:")
    for state, action in list(improved_policy_first_visit.items()):
        print(f"State: {state}, Best Action: {action}")

    print("\nImproved Policy from Every-Visit Monte Carlo:")
    for state, action in list(improved_policy_every_visit.items()):
        print(f"State: {state}, Best Action: {action}")

if __name__ == "__main__":
    main()
