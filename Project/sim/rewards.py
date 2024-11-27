from typing import List, Tuple
from sim.agents.agents import Agent
import numpy as np
from utils.config import Config
from utils import get_enemy_positions

config = Config('./config')


def reward_full(observations, agents, border_positions, obstacles, t):
    """
    Compute rewards for all agents, incorporating penalties for collisions with obstacles.
    :param observations: All agent positions in the environment.
    :param agents: List of agent objects.
    :param border_positions: Positions defining the board edges.
    :param obstacles: List of obstacle positions.
    :param t: Current timestep in the environment.
    :return: List of rewards for each agent.
    """
    all_winners = []
    all_rewards = []
    number_winning_predator = 0

    for idx, agent in enumerate(agents):
        # Calculate the reward for each agent based on distance, winners, etc.
        rw, is_winner, enemies_near = get_reward_agent(observations, idx, agents, t)
        if agent.type == "predator" and is_winner:
            number_winning_predator += 1

        all_winners.append(is_winner)
        all_rewards.append(rw)

    hot_walls = config.reward.hot_walls and not config.env.infinite_world

    for idx, agent in enumerate(agents):
        x, y, z = observations[3 * idx], observations[3 * idx + 1], observations[3 * idx + 2]

        # Obstacle penalty
        if (x, y) in obstacles:
            all_rewards[idx] -= 1  # Apply penalty for hitting an obstacle
        else:
            # Proximity penalty: Apply a lesser penalty if the agent is near an obstacle
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if (x + dx, y + dy) in obstacles and (dx != 0 or dy != 0):
                        all_rewards[idx] -= 2  # Apply small penalty for being near an obstacle
                        break

        # Additional predator-specific or border logic
        if all_winners[idx] and agent.type == "predator":
            all_rewards[idx] += 0.8
        if config.reward.share_wins_among_predators and number_winning_predator > 0:
            if agent.type == "predator":
                all_rewards[idx] += config.reward.reward_if_predators_win * number_winning_predator

        # Border penalty (if hot walls are enabled)
        if hot_walls:
            if x in border_positions or y in border_positions or (config.env.world_3D and z in border_positions):
                all_rewards[idx] = -1  # Border collision penalty
            elif (
                (x + 1, y) in obstacles
                or (x - 1, y) in obstacles
                or (x, y + 1) in obstacles
                or (x, y - 1) in obstacles
            ):
                all_rewards[idx] = -1  # Nearby obstacle penalty

    return all_rewards


def get_reward_agent(observations, agent_index, agents: List[Agent], t):
    """
    Reward for 1 action and 1 agent at time t
    :param observations:
    :param agent_index:
    :param agents:
    :param t:
    :return: reward value
    """
    x = observations[3 * agent_index]
    y = observations[3 * agent_index + 1]
    z = observations[3 * agent_index + 2]
    min_distance = None
    agent_reward = None
    winner = False
    agent_type = agents[agent_index].type
    enemies_near = []

    # For all enemies, we compute the distance between actual agent and enemy agents
    for x_1, y_1, z_1, k in get_enemy_positions(agent_index, agents, observations):
        reward, distance = distance_reward(agent_type, x, y, z, x_1, y_1, z_1)
        # If distance is smaller...
        if min_distance is None or distance < min_distance:
            min_distance, agent_reward = distance, reward

        if ((agent_type == "predator" and distance < 1 / config.env.board_size) or
                (agent_type == "prey" and distance >= 1 / config.env.board_size)):
            enemies_near.append(k)

    # Determine if it's a winner
    if ((agent_type == "predator" and min_distance < 1 / config.env.board_size) or
            (agent_type == "prey" and min_distance >= 1 / config.env.board_size)):
        winner = True

    return agent_reward, winner, enemies_near


def distance_reward(agent_type, x, y, z, x1, y1, z1) -> Tuple:
    """
    Returns the reward obtained according to the type and the position of two agents
    Args:
        agent_type: between prey and predator
        x: position x current agent
        y: position y current agent
        z: position y current agent
        x1: position secondary agent
        y1: position secondary agent
        z1: position secondary agent

    Returns: (reward, distance_between_agents)
    """
    assert agent_type in ['prey', 'predator'], "Agent type is not correct."
    dx = x - x1
    dy = y - y1
    dz = z - z1
    if config.env.infinite_world:
        dx = min(abs(dx), abs(x - x1 - 1), abs(x - x1 + 1))
        dy = min(abs(dy), abs(y - y1 - 1), abs(y - y1 + 1))
        dz = min(abs(dz), abs(z - z1 - 1), abs(z - z1 + 1))
    return distance_reward_prey(dx, dy, dz) if agent_type == "prey" else distance_reward_predator(dx, dy, dz)


def distance_reward_prey(dx, dy, dz):
    """
    Returns: Reward = $1 - \exp(-c \cdot d^2)$
    """
    distance = np.linalg.norm([dx, dy, dz])
    rw = 1 - 2 * np.exp(-config.reward.coef_distance_reward_prey * distance * distance)
    return rw, distance


def distance_reward_predator(dx, dy, dz):
    """
    Returns: Reward = $\exp(-c \cdot d^2)$
    """
    distance = np.linalg.norm([dx, dy, dz])
    rw = np.exp(-config.reward.coef_distance_reward_predator * distance * distance)
    return rw, distance

