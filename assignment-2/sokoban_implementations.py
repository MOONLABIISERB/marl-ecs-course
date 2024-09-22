import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

class SokobanEnvironment:
    def __init__(self):
        self.grid = [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 1],
            [1, 3, 0, 0, 0, 1],
            [1, 0, 0, 2, 0, 1],
            [1, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]
        self.agent_position = (1, 2)  # Agent starts at (1, 2)
        self.box_positions = [(4, 3)]
        self.storage_positions = [(3, 1)]
        self.actions = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }
        self.done = False

    def display_grid(self):
        color_grid = np.full((7, 6, 3), fill_value=[0.5, 0.5, 0.5])  # Default gray for the entire grid
        colors = {
            0: [0.8, 0.8, 0.8],  # Floor - light gray
            1: [0, 0, 0],        # Wall - black
            2: [0, 0, 1],        # Box - blue
            3: [0, 1, 0],        # Storage - green
            4: [1, 0, 0]         # Agent - red
        }
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                cell_value = self.grid[row][col]
                color_grid[row, col] = colors.get(cell_value, [0.5, 0.5, 0.5])
        agent_x, agent_y = self.agent_position
        color_grid[agent_x, agent_y] = colors[4]
        for box in self.box_positions:
            box_x, box_y = box
            color_grid[box_x, box_y] = colors[2]
        for storage in self.storage_positions:
            storage_x, storage_y = storage
            color_grid[storage_x, storage_y] = colors[3]
        plt.imshow(color_grid, interpolation='nearest')
        plt.xticks([]), plt.yticks([])
        plt.title("Sokoban Puzzle Environment")
        plt.show()

    def reset(self):
        self.agent_position = (1, 2)
        self.box_positions = [(4, 3)]
        self.done = False

    def get_next_state(self, action):
        dx, dy = action
        x, y = self.agent_position
        next_agent_position = (x + dx, y + dy)

        if self.is_valid_position(next_agent_position):
            if next_agent_position in self.box_positions:
                next_box_position = (next_agent_position[0] + dx, next_agent_position[1] + dy)
                if self.is_valid_position(next_box_position) and next_box_position not in self.box_positions:
                    self.box_positions.remove(next_agent_position)
                    self.box_positions.append(next_box_position)
                    return next_agent_position
                else:
                    return self.agent_position
            else:
                return next_agent_position
        else:
            return self.agent_position  # If move is invalid, return the current position
        
    def is_valid_position(self, position):
        x, y = position
        if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
            return False
        if self.grid[x][y] == 1:  # Wall
            return False
        return True

    def step(self, action):
        if self.done:
            return self.agent_position, 0, self.done  # Ensure it returns values even if done

        # Move the agent
        previous_agent_position = self.agent_position
        self.agent_position = self.get_next_state(self.actions[action])

        # Check for rewards and termination conditions
        if self.agent_position != previous_agent_position:
            # If a box was pushed to a storage location
            if previous_agent_position in self.box_positions:
                box_index = self.box_positions.index(previous_agent_position)
                if self.box_positions[box_index] in self.storage_positions:
                    reward = 5  # Reward for pushing box to storage
                else:
                    reward = -1  # Box is not at storage
            else:
                reward = -1  # General step cost

            # Check if all boxes are on storage locations
            if all(box in self.storage_positions for box in self.box_positions):
                print("All boxes are placed in storage! Episode complete.")
                self.done = True

            # Check if a box gets stuck (simple check for corners)
            for box in self.box_positions:
                if self.is_box_stuck(box):
                    print("A box is stuck! Episode complete.")
                    self.done = True
        else:
            # Add penalty for hitting a wall
            if not self.is_valid_position(self.agent_position):
                reward = -10  # Penalty for hitting a wall
            else:
                reward = -1  # No movement means penalty

        return self.agent_position, reward, self.done  # Ensure this is always returned

    def is_box_stuck(self, box_position):
        x, y = box_position
        if (self.grid[x-1][y] == 1 and self.grid[x][y-1] == 1) or \
           (self.grid[x-1][y] == 1 and self.grid[x][y+1] == 1) or \
           (self.grid[x+1][y] == 1 and self.grid[x][y-1] == 1) or \
           (self.grid[x+1][y] == 1 and self.grid[x][y+1] == 1):
            return True
        return False

class SokobanAgent:
    def __init__(self, env):
        self.env = env
        self.value_function = defaultdict(float)
        # Initialize the policy for all states
        self.policy = {state: random.choice(list(env.actions.keys())) for state in self.get_all_states()}
        self.discount_factor = 0.9
        self.alpha = 0.1  # Learning rate for MC Control
        self.epsilon = 0.3  # Higher exploration rate

    def value_iteration(self, theta=1e-5):
        while True:
            delta = 0
            for state in self.get_all_states():
                v = self.value_function[state]
                self.value_function[state] = max(self.calculate_q_value(state, action) for action in self.env.actions.keys())
                delta = max(delta, abs(v - self.value_function[state]))
            if delta < theta:
                break

        self.extract_optimal_policy()

    def calculate_q_value(self, state, action):
        total = 0
        for next_action in self.env.actions.keys():
            next_state, reward, _ = self.simulate_action(state, action)
            total += (1 / len(self.env.actions)) * (reward + self.discount_factor * self.value_function[next_state])
        return total

    def simulate_action(self, state, action):
        x, y = state
        dx, dy = self.env.actions[action]
        next_agent_position = (x + dx, y + dy)

        if self.env.is_valid_position(next_agent_position):
            if next_agent_position in self.env.box_positions:
                next_box_position = (next_agent_position[0] + dx, next_agent_position[1] + dy)
                if self.env.is_valid_position(next_box_position) and next_box_position not in self.env.box_positions:
                    next_agent_position = next_box_position
                    return next_agent_position, 5, False  # Reward for pushing box to storage
            return next_agent_position, -1, False  # Reward for step, not terminal
        return state, -10, False  # Penalty for hitting a wall

    def get_all_states(self):
        states = []
        for x in range(len(self.env.grid)):
            for y in range(len(self.env.grid[0])):
                states.append((x, y))
        return states

    def extract_optimal_policy(self):
        for state in self.get_all_states():
            if state in self.value_function:  # Ensure the state has a value
                self.policy[state] = max(self.env.actions.keys(), key=lambda action: self.calculate_q_value(state, action))

    def monte_carlo_control(self, num_episodes):
        returns = defaultdict(list)
        for episode_num in range(num_episodes):
            episode_data = self.generate_episode()
            visited_states = set()
            G = 0

            for state, action, reward in reversed(episode_data):
                G = reward + self.discount_factor * G
                if state not in visited_states:
                    returns[(state, action)].append(G)
                    self.value_function[state] = np.mean(returns[(state, action)])
                    visited_states.add(state)
            self.extract_optimal_policy()  # Update policy after each episode

    def generate_episode(self):
        episode = []
        state = self.env.agent_position
        done = False

        while not done:
            # Check if the state is in the policy before accessing it
            if state in self.policy:
                if random.random() < self.epsilon:  # Explore
                    action = random.choice(list(self.env.actions.keys()))
                else:  # Exploit
                    action = self.policy[state]
            else:
                action = random.choice(list(self.env.actions.keys()))  # Default to random action if state not in policy

            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

if __name__ == "__main__":
    env = SokobanEnvironment()
    env.display_grid()

    # Run Dynamic Programming
    agent_dp = SokobanAgent(env)
    agent_dp.value_iteration()
    print("Value Function after Value Iteration:")
    print("{:<15} {:<15}".format("State", "Value"))
    print("-" * 30)
    for state, value in agent_dp.value_function.items():
        print(f"{str(state):<15} {value:<15.5f}")
    
    print("\nOptimal Policy after Value Iteration:")
    print("{:<15} {:<15}".format("State", "Optimal Action"))
    print("-" * 30)
    for state, action in agent_dp.policy.items():
        print(f"{str(state):<15} {action:<15}")

    # Reset environment for Monte Carlo
    env.reset()

    # Run Monte Carlo Control
    agent_mc = SokobanAgent(env)
    agent_mc.monte_carlo_control(num_episodes=1000)
    print("\nValue Function after Monte Carlo Control:")
    print("{:<15} {:<15}".format("State", "Value"))
    print("-" * 30)
    for state, value in agent_mc.value_function.items():
        print(f"{str(state):<15} {value:<15.5f}")

    print("\nOptimal Policy after Monte Carlo Control:")
    print("{:<15} {:<15}".format("State", "Optimal Action"))
    print("-" * 30)
    for state, action in agent_mc.policy.items():
        print(f"{str(state):<15} {action:<15}")
