from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from sim.rewards import reward_full
from sim.agents.agents import Agent
import random


class Env:
    agents: List[Agent]

    def __init__(self, env_config, config):
        self.reward_type = env_config.reward_type
        self.noise = env_config.noise
        self.board_size = env_config.board_size

        self.plot_radius = env_config.plot_radius

        self.possible_location_values = [float(k) / float(self.board_size) for k in range(self.board_size)]

        self.current_iteration = 0
        self.max_iterations = env_config.max_iterations

        self.infinite_world = env_config.infinite_world
        self.config = config

        self.obstacles = env_config.obstacles  # List of obstacle coordinates (x, y)

        self.magic_switch = None
        self.initial_types = []

        self.obstacle_positions = []
        for (x, y) in self.obstacles:
            self.obstacle_positions.append(self.possible_location_values[x])
            self.obstacle_positions.append(self.possible_location_values[y])

        self.agents = []
        self.initial_positions = []

    def add_agent(self, agent: Agent, position=None):
        """
        Add an agent to the environment, either at a specified position or a random one.
        """
        assert position is None or (0 <= position[0] < 1 and 0 <= position[1] < 1), "Initial position is incorrect."
        if self.config.env.world_3D:
            assert position is None or len(position) == 3, "Please provide 3D positions if using a 3D world."
        if position is not None:
            assert position not in self.obstacles, "Initial position in an obstacle"
        if position is not None:
            x, y = self.possible_location_values[-1], self.possible_location_values[-1]
            z = position[2] if len(position) == 3 else 0
            for k in range(self.board_size):
                if position[0] <= self.possible_location_values[k]:
                    x = self.possible_location_values[k]
                if position[1] <= self.possible_location_values[k]:
                    y = self.possible_location_values[k]
                if len(position) == 2 and position[2] <= self.possible_location_values[k]:
                    z = self.possible_location_values[k]
            position = x, y, z
        self.agents.append(agent)
        self.initial_types.append(agent.type)
        self.initial_positions.append(position)

    def _get_random_position(self):
        """
        Get a random valid position that is not occupied by obstacles.
        """
        possible_values = [(x, y) for x in range(self.board_size)
                           for y in range(self.board_size) if (x, y) not in self.obstacles]
        x, y = random.sample(possible_values, 1)[0]
        x, y = self.possible_location_values[x], self.possible_location_values[y]
        z = self.possible_location_values[0]
        if self.config.env.world_3D:
            z = random.sample(self.possible_location_values, 1)[0]
        return x, y, z

    def _get_position_from_action(self, current_position, action):
        """
        From an action number, returns the new position. Checks if the new position is valid.
        """
        index_x = self.possible_location_values.index(current_position[0])
        index_y = self.possible_location_values.index(current_position[1])
        index_z = 0
        if self.config.env.world_3D:
            index_z = self.possible_location_values.index(current_position[2])

        if action == 1:  # Front
            position = index_x, index_y + 1, index_z
        elif action == 2:  # Left
            position = index_x - 1, index_y, index_z
        elif action == 3:  # Back
            position = index_x, index_y - 1, index_z
        elif action == 4:  # Right
            position = index_x + 1, index_y, index_z
        elif self.config.env.world_3D and action == 5:  # Top
            position = index_x, index_y, index_z + 1
        elif self.config.env.world_3D and action == 6:  # Bottom
            position = index_x, index_y, index_z - 1
        else:  # None (=0)
            position = index_x, index_y, index_z
        
        if not self.infinite_world:
            if position[0] < 0 or position[0] >= len(self.possible_location_values):
                position = index_x, position[1], position[2]
            if position[1] < 0 or position[1] >= len(self.possible_location_values):
                position = position[0], index_y, position[2]
            if position[2] < 0 or position[2] >= len(self.possible_location_values):
                position = position[0], position[1], index_z
        else:
            position = (position[0] % len(self.possible_location_values),
                        position[1] % len(self.possible_location_values),
                        position[2] % len(self.possible_location_values))

        # Check if new position collides with obstacles
        if [position[0], position[1]] in self.obstacles:
            return current_position  # Return current position if there's an obstacle

        position = (self.possible_location_values[position[0]], 
                    self.possible_location_values[position[1]], 
                    self.possible_location_values[position[2]])

        # Check for magic switch
        if self.config.env.magic_switch and position[0] == self.magic_switch[0] and position[1] == self.magic_switch[1]:
            for agent in self.agents:
                if agent.type == "predator":
                    agent.type = "prey"
                else:
                    agent.type = "predator"

        return position

    def _get_state_from_positions(self, positions):
        """
        Get the state representation from the agent positions.
        """
        states = []
        for k in range(len(self.agents)):
            state = positions[:]
            state.extend(self.obstacle_positions)
            if self.config.env.magic_switch:
                state.extend([self.magic_switch[0], self.magic_switch[1]])
                types = [int(agent.type == "predator") for agent in self.agents]
                state.extend(types)
            states.append(state)
        return states
    
    def _get_possible_positions(self, current_position):
        # """
        # Return possible positions from the given one
        # Args:
        #     current_position:
        # Returns: x_index, y_index of the possible new positions
        # """
        index_x = self.possible_location_values.index(current_position[0])
        index_y = self.possible_location_values.index(current_position[1])
        index_z = self.possible_location_values.index(current_position[2])
        max_len = len(self.possible_location_values)
        indexes = [(index_x, index_y, index_z)]
        if (self.infinite_world or index_x > 0) and ((index_x - 1) % max_len, index_y) not in self.obstacles:
            indexes.append(((index_x - 1) % max_len, index_y, index_z))  # Left
        if (self.infinite_world or index_x < len(self.possible_location_values) - 1) and (
                ((index_x + 1) % max_len, index_y) not in self.obstacles):  # Right
            indexes.append(((index_x + 1) % max_len, index_y, index_z))
        if (self.infinite_world or index_y > 0) and (
                (index_x, (index_y - 1) % max_len) not in self.obstacles):  # Back
            indexes.append((index_x, (index_y - 1) % max_len, index_z))
        if (self.infinite_world or index_y < len(self.possible_location_values) - 1) and (
                (index_x, (index_y + 1) % max_len) not in self.obstacles):  # Front
            indexes.append((index_x, (index_y + 1) % max_len, index_z))
        if self.config.env.world_3D:
            if (self.infinite_world or index_z < len(self.possible_location_values) - 1) and (
                    (index_x, index_y) not in self.obstacles):  # Top
                indexes.append((index_x, index_y, (index_z + 1) % max_len))
            if (self.infinite_world or index_z > 0) and (
                    (index_x, index_y) not in self.obstacles):  # Bottom
                indexes.append((index_x, index_y, (index_z - 1) % max_len))
        return indexes

    def _get_collisions(self, positions):
        n_collisions = 0
        for i, agent in enumerate(self.agents):
            x, y, z = positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]
            for j, other_agent in enumerate(self.agents):
                x_2, y_2, z_2 = positions[3 * j], positions[3 * j + 1], positions[3 * j + 2]
                distance = np.linalg.norm([x_2 - x, y_2 - y, z_2 - z])
                if agent.type != other_agent.type and distance < self.possible_location_values[1]:
                    n_collisions += 1
        return n_collisions // 2

    def reset(self, test=False):
        """
        Reset the environment and return initial state for each agent.
        """
        self.current_iteration = 0
        absolute_positions = []
        for k in range(len(self.initial_positions)):
            position = self.initial_positions[k]
            if position is None:  # If random position
                position = self._get_random_position()
            absolute_positions.append(position[0])
            absolute_positions.append(position[1])
            absolute_positions.append(position[2])
        types = [agent.type for agent in self.agents]
        if self.config.env.magic_switch:
            self.magic_switch = self._get_random_position()[:2]
            for k in range(len(self.agents)):
                if not test:
                    self.agents[k].type = "predator" if self.agents[k].type == "prey" else "prey"
                else:
                    self.agents[k].type = self.initial_types[k]
                types[k] = self.agents[k].type
        return self._get_state_from_positions(absolute_positions), types

    def step(self, prev_states, actions):
        """
        Step through the environment: move agents, calculate rewards, handle penalties, etc.
        """
        positions = []
        penalties = 0  # Track penalty for collisions
        for k in range(len(self.agents)):
            position = prev_states[0][3 * k], prev_states[0][3 * k + 1], prev_states[0][3 * k + 2]
            new_position = self._get_position_from_action(position, actions[k])
            
            # If the agent is colliding with an obstacle, apply a penalty (negative reward)
            if new_position == position:  # Same position means collision with obstacle
                penalties += 1  # Increment penalty
        
            positions.append(new_position[0])
            positions.append(new_position[1])
            positions.append(new_position[2])

        n_collisions = self._get_collisions(positions)
        next_state = self._get_state_from_positions(positions)
        # Calculate rewards and apply penalty if needed
        border_positions = [self.possible_location_values[0], self.possible_location_values[-1]]
        rewards = reward_full(positions, self.agents, border_positions, self.obstacles, self.current_iteration)
        if penalties > 0:
            rewards = [reward - 5 * penalties for reward in rewards]  # Apply a penalty to the rewards

        types = [agent.type for agent in self.agents]
        self.current_iteration += 1
        terminal = False
        if self.current_iteration == self.max_iterations:
            terminal = True
        
        return next_state, rewards, terminal, n_collisions, types

    def plot(self, state, types, rewards, ax):
        """
        Plot the environment and agents, including obstacles and magic switch.
        """
        tick_labels = np.arange(0, self.board_size)
        ax.set_xticks(self.possible_location_values)
        ax.set_yticks(self.possible_location_values)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.set_xlim(0, self.possible_location_values[-1])
        ax.set_ylim(0, self.possible_location_values[-1])
        if self.config.env.world_3D:
            ax.set_zticks(self.possible_location_values)
            ax.set_zticklabels(tick_labels)
            ax.set_zlim(0, self.possible_location_values[-1])
        ax.grid(which="major", alpha=0.5)
        
        # Plot obstacles
        for x, y in self.obstacles:
            x, y = self.possible_location_values[x], self.possible_location_values[y]
            side = self.possible_location_values[1]
            if self.config.env.world_3D:
                top = self.possible_location_values[-1]
                points = [(x - side / 2, y - side / 2, 0), (x - side / 2, y + side / 2, 0),
                          (x + side / 2, y + side / 2, 0),
                          (x + side / 2, y - side / 2, 0),
                          (x - side / 2, y - side / 2, top), (x - side / 2, y + side / 2, top),
                          (x + side / 2, y + side / 2, top), (x + side / 2, y - side / 2, top)]
                edges = [[points[0], points[1], points[2], points[3]],
                         [points[4], points[5], points[6], points[7]],
                         [points[0], points[1], points[5], points[4]],
                         [points[2], points[3], points[6], points[7]],
                         [points[0], points[4], points[7], points[3]],
                         [points[1], points[5], points[6], points[2]]]
                block = Poly3DCollection(edges, linewidth=0)
                block.set_facecolor((0, 0, 0, 0.1))
                ax.add_collection3d(block)
            else:
                block = plt.Rectangle((x - side / 2, y - side / 2), width=side, height=side, linewidth=0, color="black")
                ax.add_patch(block)

        # Plot magic switch
        if self.config.env.magic_switch:
            x, y = self.magic_switch
            side = self.possible_location_values[1]
            block = plt.Rectangle((x - side / 2, y - side / 2), width=side, height=side, linewidth=0, color="purple")
            ax.add_patch(block)

        # Plot agents
        for k in range(len(self.agents)):
            if self.config.env.world_3D:
                position = state[0][3 * k], state[0][3 * k + 1], state[0][3 * k + 2]
            else:
                position = state[0][3 * k], state[0][3 * k + 1]
            radius = self.config.env.plot_radius_3D if self.config.env.world_3D else self.plot_radius
            self.agents[k].plot(position, types[k], rewards[k], radius, ax)
