import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from environment import MultiAgentShipTowEnv
from maddpg import MADDPGAgent


def load_best_model(env, model_dir='saved_models') -> list:
    """
    Load the most recently trained model from the saved_models directory.

    Args:
        env (MultiAgentShipTowEnv): environment object
        model_dir (str): directory containing the saved models

    Returns:
        agents (list): list of agents with trained models
    """
    actor_files = [f for f in os.listdir(model_dir) if f.endswith('_actor_maddpg.pth')]
    
    if not actor_files:
        raise FileNotFoundError("No trained models found in the saved_models directory.")
    
    state_size = env.observation_space['tugboat_1'].shape[0]
    action_size = env.action_space['tugboat_1'].shape[0]
    
    # Create agents
    agents = []
    for i in range(2):
        agent = MADDPGAgent(state_size, action_size)
        agent.actor.load_state_dict(torch.load(f'saved_models/tugboat_{i+1}_actor_maddpg.pth'))
        agents.append(agent)
    
    return agents



def test_and_visualize_trajectory(env, agents, max_steps=1000000, save_path='trajectory_visualization.png') -> tuple:
    """
    Test the trained model and create a detailed trajectory visualization.

    Args:
        env (MultiAgentShipTowEnv): environment object
        agents (list): list of agents with trained models
        max_steps (int): maximum number of steps to run the simulation
        save_path (str): path to save the trajectory visualization

    Returns:
        tuple: ship, tugboat 1, and tugboat 2 trajectories
    """
    observations = env.reset()
    
    # Trajectory tracking
    ship_trajectory = []
    tugboat1_trajectory = []
    tugboat2_trajectory = []
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        actions = {}
        for i, agent_id in enumerate(['tugboat_1', 'tugboat_2']):
            obs = torch.FloatTensor(observations[agent_id])
            
            # Get action from actor network
            with torch.no_grad():
                action = agents[i].actor(obs).numpy()
                actions[agent_id] = np.clip(action, 0, 10)
        
        next_observations, rewards, dones, _ = env.step(actions)
        done = dones['__all__']
        
        # Track trajectories
        ship_trajectory.append((observations['tugboat_1'][0], observations['tugboat_1'][1]))
        tugboat1_trajectory.append((observations['tugboat_1'][3], observations['tugboat_1'][4]))
        tugboat2_trajectory.append((observations['tugboat_1'][6], observations['tugboat_1'][7]))
        
        observations = next_observations
        step += 1
    
    plt.figure(figsize=(12, 10))
    plt.title('Ship and Tugboats Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    plt.xlim(0, env.grid_size)
    plt.ylim(0, env.grid_size)

    # Plot dock
    dock_rect = patches.Rectangle(
        env.target_position, 
        env.dock_dim[0], 
        env.dock_dim[1], 
        facecolor='brown', 
        alpha=0.5
    )
    plt.gca().add_patch(dock_rect)
    
    # Plot obstacles
    for obstacle in env.obstacles:
        obs_rect = patches.Rectangle(
            (obstacle[0], obstacle[1]), 
            obstacle[2], 
            obstacle[3], 
            facecolor='red', 
            alpha=0.3
        )
        plt.gca().add_patch(obs_rect)
    
    # Plot trajectories
    plt.plot(*zip(*ship_trajectory), 'b-', label='Ship Path', linewidth=2)
    plt.plot(*zip(*tugboat1_trajectory), 'g--', label='Tugboat 1 Path', linewidth=1.5)
    plt.plot(*zip(*tugboat2_trajectory), 'r--', label='Tugboat 2 Path', linewidth=1.5)
    
    # Starting and ending points
    plt.scatter(ship_trajectory[0][0], ship_trajectory[0][1], color='green', s=100, label='Start')
    plt.scatter(ship_trajectory[-1][0], ship_trajectory[-1][1], color='red', s=100, label='End')
    
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()
    
    print(f"Trajectory visualization saved to {save_path}")
    print(f"Total steps: {step}")
    
    return ship_trajectory, tugboat1_trajectory, tugboat2_trajectory


def main():
    env = MultiAgentShipTowEnv()
    
    try:
        agents = load_best_model(env)
    except FileNotFoundError as e:
        print(e)
        return
    
    trajectories = test_and_visualize_trajectory(env, agents)

if __name__ == '__main__':
    main()