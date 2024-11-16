import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import time
from environment import MAPFEnv

class Agent:
    def __init__(self, num_actions=5, alpha=0.1, gamma=0.995, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.num_actions))
        return np.argmax(self.q_table.get(state, np.zeros(self.num_actions)))

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)
        
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error


class MARLTrainer:
    def __init__(self, env, num_episodes=1000, max_steps_per_episode=100):
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.agents = {agent_id: Agent() for agent_id in env.agents}
        self.logs = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_facecolor('black')

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_reward = {agent_id: 0 for agent_id in self.env.agents}
            episode_log = {"episode": episode + 1, "steps": [], "total_rewards": total_reward.copy()}
            
            print(f"\nStarting Episode {episode + 1}/{self.num_episodes}")

            for step in range(self.max_steps_per_episode):
                print(f"  Step {step + 1}: Agent Actions and Positions")
                actions = {agent_id: self.agents[agent_id].choose_action(self.get_state_repr(state[agent_id])) for agent_id in state}
                next_state, rewards, done = self.env.step(actions)
                self.visualize_movement()
                
                step_log = {"step": step + 1, "actions": actions, "positions": self.env.positions.copy(), "rewards": rewards.copy()}
                episode_log["steps"].append(step_log)

                for agent_id, agent in self.agents.items():
                    agent_state = self.get_state_repr(state[agent_id])
                    next_agent_state = self.get_state_repr(next_state[agent_id])
                    agent.update_q_value(agent_state, actions[agent_id], rewards[agent_id], next_agent_state)
                    total_reward[agent_id] += rewards[agent_id]

                state = next_state
                
                if done["__all__"]:
                    print("  All agents reached their goals.")
                    break
        
        self.save_q_tables()
        self.save_logs()
        plt.ioff()
        plt.close(self.fig) 

    def get_state_repr(self, observation):
        return tuple(observation.flatten())

    def save_q_tables(self, filename="q_tables_env.pkl"):
        q_tables = {agent_id: agent.q_table for agent_id, agent in self.agents.items()}
        with open(filename, "wb") as f:
            pickle.dump(q_tables, f)
        print("Q-tables saved to", filename)

    def save_logs(self, filename="training_logs_env.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.logs, f)
        print("Training logs saved to", filename)

    def visualize_movement(self):
        """
        Visualizes the current state of the environment with all agents, obstacles, and goals.
        Updates a single figure without creating multiple plots.
        """
        self.ax.clear()
        grid = np.zeros(self.env.grid_size)
        for obs in self.env.obstacles:
            self.ax.add_patch(plt.Rectangle((obs[0], obs[1]), 1, 1, color='grey'))
        for agent, pos in self.env.positions.items():
            goal = self.env.goals[agent]
            self.ax.add_patch(plt.Rectangle((pos[0], pos[1]), 1, 1, color=self.env.agents[agent]['color']))
            self.ax.text(goal[0] +0.5 , goal[1] +0.5, '+', ha='center', va='center', color=self.env.agents[agent]['color'])
        self.ax.set_xticks(np.arange(0, self.env.grid_size[1] + 1, 1))
        self.ax.set_yticks(np.arange(0, self.env.grid_size[0] + 1, 1))
        self.ax.grid(color='gray')
        self.ax.set_xlim(0, self.env.grid_size[1])
        self.ax.set_ylim(0, self.env.grid_size[0])
        self.ax.set_aspect('equal')
        plt.draw()
        plt.pause(0.5)
if __name__ == "__main__":
    grid_size = (10, 10)
    num_agents = 4
    num_episodes = 500

    env = MAPFEnv(grid_size=grid_size, num_agents=num_agents)
    trainer = MARLTrainer(env, num_episodes=num_episodes, max_steps_per_episode=100)
    
    trainer.train()
