import random
from collections import defaultdict
import numpy as np

class SokobanAgent:
    def __init__(self, env):
        self.env = env
        self.value_function = defaultdict(float)
        self.policy = {state: random.choice(list(env.actions.keys())) for state in self.get_all_states()}
        self.discount_factor = 0.9
        self.alpha = 0.1 
        self.epsilon = 0.3  

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
                    return next_agent_position, 5, False  
            return next_agent_position, -1, False 
        return state, -10, False  

    def get_all_states(self):
        states = []
        for x in range(len(self.env.grid)):
            for y in range(len(self.env.grid[0])):
                states.append((x, y))
        return states

    def extract_optimal_policy(self):
        for state in self.get_all_states():
            if state in self.value_function:  
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
            self.extract_optimal_policy()  

    def generate_episode(self):
        episode = []
        state = self.env.agent_position
        done = False

        while not done:
            if state in self.policy:
                if random.random() < self.epsilon:  
                    action = random.choice(list(self.env.actions.keys()))
                else:  
                    action = self.policy[state]
            else:
                action = random.choice(list(self.env.actions.keys()))  

            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode
