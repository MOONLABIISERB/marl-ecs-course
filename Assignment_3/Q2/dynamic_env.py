import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from heapq import heappop, heappush
import random 


class MAPFEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), num_agents=4):
        super(MAPFEnv, self).__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size[0], self.grid_size[1], self.num_agents), dtype=np.float32)
        self.action_dict = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        self.obstacles = [(5, 0), (5, 1), (5, 2), (4, 2), (0, 5), (1, 5), (2, 5), (2, 4), (4, 9), (4, 8), (4, 7), (5, 7), (9, 5), (8, 5), (7, 5), (7, 4)]
        self.agents = {
            '1': {'start': (), 'goal': (5,8), 'color': 'cyan'},
            '2': {'start': (), 'goal': (8, 4), 'color': 'green'},
            '3': {'start': (), 'goal': (4, 1), 'color': '#800080'},
            '4': {'start': (), 'goal': (1, 4), 'color': 'yellow'}
        }
        self.start_positions = self.initialize_random_positions()
        self.penalty = -1
        self.goal_reward = 10 
        self.use_a_star = False
        for agent,pos in self.start_positions.items():
            self.agents[agent]['start'] = pos
        self.reset()
    def initialize_random_positions(self):
        """
        Initializes random start positions for agents within the grid, ensuring that
        start points do not collide with goals, obstacles, or other agents.
        """
        self.goals = {}
        for agent,pos in self.agents.items():
            self.goals[agent] = pos['goal']
        occupied_positions = set(self.obstacles) | set(self.goals.values())
        positions = {}

        for agent in self.goals.keys():
            while True:
                start_x = random.randint(0, self.grid_size[0] - 1)
                start_y = random.randint(0, self.grid_size[1] - 1)
                start_pos = (start_x, start_y)
                if start_pos not in occupied_positions:
                    positions[agent] = start_pos
                    occupied_positions.add(start_pos)
                    break

        return positions

    def reset(self):
        self.positions = {agent: self.agents[agent]['start'] for agent in self.agents}
        self.goals = {agent: self.agents[agent]['goal'] for agent in self.agents}
        if self.use_a_star:
            self.paths = {agent: self.find_path(self.positions[agent], self.goals[agent]) for agent in self.agents}
        else:
            self.paths = None
        
        return {agent: self._get_observation(agent) for agent in self.agents}

    def find_path(self, start, goal):
        """ Implements A* or Dijkstra's algorithm """
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in self.action_dict.values():
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < self.grid_size[0] and 
                    0 <= neighbor[1] < self.grid_size[1] and 
                    neighbor not in self.obstacles):
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heappush(open_set, (f_score, neighbor))

        return [start] 

    def step(self, actions=None):
        rewards = {}
        obs = {}
        done = {}

        if self.use_a_star and self.paths: 
            new_positions = {}
            for agent, path in self.paths.items():
                current_pos = self.positions[agent]
                next_pos = path.pop(0) if path else current_pos
                new_positions[agent] = next_pos if next_pos not in self.obstacles else current_pos
            final_positions = self._resolve_collisions(new_positions)
        else: 
            final_positions = {}
            intended_positions = {}
            for agent, action in actions.items():
                current_pos = self.positions[agent]
                dx, dy = self.action_dict[action]
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                intended_positions[agent] = next_pos if (0 <= next_pos[0] < self.grid_size[0] and 
                                                          0 <= next_pos[1] < self.grid_size[1] and
                                                          next_pos not in self.obstacles) else current_pos

            final_positions = self._resolve_collisions(intended_positions)

        all_reached_goals = True
        for agent, new_pos in final_positions.items():
            self.positions[agent] = new_pos
            if new_pos == self.goals[agent]:
                rewards[agent] = self.goal_reward
                done[agent] = True
            else:
                rewards[agent] = self.penalty
                done[agent] = False
                all_reached_goals = False

            obs[agent] = self._get_observation(agent)

        done["__all__"] = all_reached_goals
        return obs, rewards, done

    def _resolve_collisions(self, intended_positions):
        final_positions = {}

        cell_counts = {}
        for agent, pos in intended_positions.items():
            if pos in cell_counts:
                cell_counts[pos].append(agent)
            else:
                cell_counts[pos] = [agent]
        for agent, intended_pos in intended_positions.items():
            if len(cell_counts[intended_pos]) > 1:
                final_positions[agent] = self.positions[agent]
            else:
                final_positions[agent] = intended_pos

        return final_positions

    def _get_observation(self, agent):
        observation = np.zeros(self.grid_size, dtype=np.float32)
        pos = self.positions[agent]
        observation[pos[0], pos[1]] = 1
        return observation
    
    def render(self):
        grid = np.zeros((self.grid_size[0],self.grid_size[1]))
        fig,ax = plt.subplots()
        for obs in self.obstacles:
            ax.add_patch(plt.Rectangle((obs[0], obs[1]), 1, 1, color='black'))
        for agent,markings in self.agents.items():
            pos = markings['start']
            goal = markings['goal']
            ax.add_patch(plt.Rectangle((pos[0], pos[1]), 1, 1, color=self.agents[agent]['color']))
            ax.text(pos[0] + 0.5, pos[1] + 0.5, agent, ha='center', va='center', color='black')
            ax.text(goal[0] + 0.5, goal[1] + 0.5, '+', ha='center', va='center', color=self.agents[agent]['color'])
        ax.set_xticks(np.arange(0, self.grid_size[1] + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size[0] + 1, 1))
        ax.grid(color='gray')
        ax.set_xlim(0, self.grid_size[1])
        ax.set_ylim(0, self.grid_size[0])
        ax.set_aspect('equal')
        plt.draw()
        plt.show()

def synchronized_agent_movement(env, paths, delay=0.5):
    """
    Moves all agents in a synchronized manner along their precomputed paths, displays the grid,
    and returns the time taken by each agent to reach their goal, along with a ranked summary.
    """
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    all_reached_goals = False
    step_counter = 0
    completion_times = {agent: None for agent in env.agents}

    while not all_reached_goals:
        ax.clear()
        step_counter += 1
        grid = np.zeros(env.grid_size)
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle((obs[1], obs[0]), 1, 1, color='grey'))
        all_reached_goals = True
        for agent, path in paths.items():
            if path and completion_times[agent] is None: 
                next_pos = path.pop(0)
                env.positions[agent] = next_pos
                if next_pos == env.goals[agent]:
                    completion_times[agent] = step_counter

            pos = env.positions[agent]
            goal = env.goals[agent]
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, color=env.agents[agent]['color']))
            ax.text(pos[1] + 0.5, pos[0] + 0.5, agent, ha='center', va='center', color='black')
            ax.text(goal[1] + 0.5, goal[0] + 0.5, '+', ha='center', va='center', color=env.agents[agent]['color'])

            if completion_times[agent] is None:
                all_reached_goals = False

        ax.set_xticks(np.arange(0, env.grid_size[1] + 1, 1))
        ax.set_yticks(np.arange(0, env.grid_size[0] + 1, 1))
        ax.grid(color='gray')
        ax.set_xlim(0, env.grid_size[1])
        ax.set_ylim(0, env.grid_size[0])
        ax.set_aspect('equal')
        plt.draw()
        plt.pause(delay)

    plt.ioff()
    plt.show()

    rankings = sorted(completion_times.items(), key=lambda x: x[1] if x[1] is not None else float('inf'))
    return completion_times, rankings


env = MAPFEnv()
env.render()
