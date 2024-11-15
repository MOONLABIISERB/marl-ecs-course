import pickle
from env import MAPFEnv
import matplotlib.pyplot as plt
import numpy as np
import time

class Agent:
    def __init__(self, q_table=None, num_actions=5, epsilon=0.0):
        self.q_table = q_table if q_table is not None else {}
        self.num_actions = num_actions
        self.epsilon = epsilon 

    def choose_action(self, state):
        if state in self.q_table:
            return np.argmax(self.q_table[state])
        return 4 

def load_agents(q_table_file):
    with open(q_table_file, "rb") as f:
        q_tables = pickle.load(f)
    return {agent_id: Agent(q_table=q_tables[agent_id]) for agent_id in q_tables}

def evaluate_agents(env, agents, max_steps=100):
    """
    agents in the environment with their loaded Q-tables and records the time taken by each agent to reach their goals are evaluated .
    """
    state = env.reset()
    total_rewards = {agent_id: 0 for agent_id in agents}
    completion_times = {agent_id: None for agent_id in agents}
    start_times = {agent_id: time.time() for agent_id in agents}

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    for step in range(max_steps):
        actions = {agent_id: agents[agent_id].choose_action(get_state_repr(state[agent_id])) for agent_id in state}
        next_state, rewards, done= env.step(actions)
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward
            if done[agent_id] and completion_times[agent_id] is None:  
                completion_times[agent_id] = time.time() - start_times[agent_id]
        visualize_movement(env, ax)
        state = next_state

        if done["__all__"]:
            print("All agents reached their goals.")
            break

    plt.ioff()
    plt.show()
    ranked_times = sorted(completion_times.items(), key=lambda x: x[1] if x[1] is not None else float('inf'))
    print("Total rewards:", total_rewards)
    print("Completion Times (in seconds):", completion_times)
    print("\nRanking of Agents by Time Taken:")
    for rank, (agent, time_taken) in enumerate(ranked_times, start=1):
        print(f"Rank {rank}: {agent} with {time_taken:.2f} seconds")

def get_state_repr(observation):
    return tuple(observation.flatten())

def visualize_movement(env, ax):
    """
    Visualizes the current state of the environment with all agents, obstacles, and goals.
    Updates a single figure without creating multiple plots.
    """
    ax.clear()
    grid = np.zeros(env.grid_size)
    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle((obs[0], obs[1]), 1, 1, color='grey'))
    for agent, pos in env.positions.items():
        goal = env.goals[agent]
        ax.add_patch(plt.Rectangle((pos[0], pos[1]), 1, 1, color=env.agents[agent]['color']))
        ax.text(goal[0] + 0.5, goal[1] + 0.5, '+', ha='center', va='center', color=env.agents[agent]['color'])
    ax.set_xticks(np.arange(0, env.grid_size[1] + 1, 1))
    ax.set_yticks(np.arange(0, env.grid_size[0] + 1, 1))
    ax.grid(color='gray')
    ax.set_xlim(0, env.grid_size[1])
    ax.set_ylim(0, env.grid_size[0])
    ax.set_aspect('equal')

    plt.draw()
    plt.pause(0.5)

if __name__ == "__main__":
    grid_size = (10, 10)
    num_agents = 4
    env = MAPFEnv(grid_size=grid_size, num_agents=num_agents)
    agents = load_agents("q_tables.pkl")
    evaluate_agents(env, agents, max_steps=100)
