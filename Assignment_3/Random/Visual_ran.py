import numpy as np
import matplotlib.pyplot as plt
from Random_env import RandomMAPFEnvironment
from Trainer_ran import RandomPositionQLearningAgent, train_random_mapf

def find_minimum_steps_to_goal(env, agents):
    """
    Find minimum steps for each agent to reach its goal in a random setup
    
    Args:
        env (RandomMAPFEnvironment): Environment instance
        agents (list): Trained Q-learning agents
    
    Returns:
        np.ndarray: Steps taken by each agent to reach goal
    """
    min_steps = np.zeros(len(agents))
    
    for agent_idx in range(len(agents)):
        current_state = env.initialize_scenario()  # Randomize starting positions
        current_position = current_state[agent_idx]
        steps = 0
        
        while current_position != env.goal_pos[agent_idx]:
            # Get action for this specific agent, keep others stationary
            action = agents[agent_idx].select_action(current_position)
            action_list = [0] * len(current_state)
            action_list[agent_idx] = action
            
            next_state, _, done = env.execute_action(current_state, action_list)
            current_position = next_state[agent_idx]
            current_state = next_state
            steps += 1
            
            if steps > 1000:  # Prevent infinite loop
                break
        
        min_steps[agent_idx] = steps
    
    return min_steps

def visualize_agent_paths(env, agents, min_steps):
    """
    Visualize paths taken by agents to reach their goals in a random setup
    
    Args:
        env (RandomMAPFEnvironment): Environment instance
        agents (list): Trained Q-learning agents
        min_steps (np.ndarray): Minimum steps taken by each agent
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set up gridlines and limits
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_xticks(np.arange(0, env.grid_size + 1, 1))
    ax.set_yticks(np.arange(0, env.grid_size + 1, 1))
    ax.grid(True, linewidth=2, color='lightgray')
    
    # Set aspect of the plot to be equal
    ax.set_aspect('equal')
    ax.set_xlabel('X-axis', fontsize=12, color='darkgray')
    ax.set_ylabel('Y-axis', fontsize=12, color='darkgray')
    
    # Remove the axes
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False, colors='darkgray')
    
    # Draw walls
    for wall in env.walls:
        ax.add_patch(plt.Rectangle((wall[0], wall[1]), 1, 1, color='dimgray'))
    
    # Initialize legend elements
    legend_elements = []

    # Tracking agent paths
    for agent_idx in range(len(agents)):
        current_state = env.initialize_scenario()  # Randomize starting positions
        current_position = current_state[agent_idx]
        path = [current_position]
        
        while current_position != env.goal_pos[agent_idx]:
            action = agents[agent_idx].select_action(current_position)
            action_list = [0] * len(current_state)
            action_list[agent_idx] = action
            next_state, _, done = env.execute_action(current_state, action_list)
            current_position = next_state[agent_idx]
            current_state = next_state
            path.append(current_position)
            
            if len(path) > 1000:  # Prevent infinite loop
                break
        
        # Plot path
        path = np.array(path)
        ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5,
                color=env.agent_colors[agent_idx],
                linewidth=2, marker='o', markersize=5, alpha=0.7)
        
        # Add legend entry for the agent with minimum steps
        legend_elements.append(
            plt.Line2D([0], [0], color=env.agent_colors[agent_idx], lw=2,
                    label=f'Agent {agent_idx + 1}: {int(min_steps[agent_idx])} steps')
        )
    
    # Add legend to the plot
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, title="Minimum Steps")
    
    # Plot initial and goal positions
    for idx, (pos, goal) in enumerate(zip(env.agent_pos, env.goal_pos)):
        # Initial position
        ax.add_patch(plt.Rectangle((pos[0], pos[1]), 1, 1, 
                    color=env.agent_colors[idx], alpha=0.8))
        ax.text(pos[0] + 0.5, pos[1] + 0.5, str(idx), 
                color='white', ha='center', va='center', fontsize=12)
        
        # Goal position
        ax.plot(goal[0] + 0.5, goal[1] + 0.5, 
                marker='+', color=env.agent_colors[idx], 
                mew=3, ms=20)
    
    plt.title('Agent Paths to Goals - Random Positions')
    plt.tight_layout()
    plt.savefig('random_mapf_agent_paths.png')
    plt.show()

if __name__ == "__main__":
    # Train agents
    trained_agents, episode_rewards, collision_counts = train_random_mapf()
    
    # Create environment
    env = RandomMAPFEnvironment()
    
    # Find minimum steps
    steps = find_minimum_steps_to_goal(env, trained_agents)
    print("Minimum steps for each agent:", steps)
    
    # Visualize paths
    visualize_agent_paths(env, trained_agents, steps)
