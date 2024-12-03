import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
from environment import MultiAgentShipTowEnv

def collect_rewards(num_episodes=100, max_steps=1000) -> tuple:
    """
    Collect rewards for tugboat 1 and tugboat 2.

    Args:
        num_episodes (int): number of episodes to collect rewards
        max_steps (int): maximum number of steps per episode

    Returns:
        tuple: rewards for tugboat 1 and tugboat 2
    """
    env = MultiAgentShipTowEnv()
    episode_rewards_tug1 = []
    episode_rewards_tug2 = []
    
    for episode in track(range(num_episodes), description="Episodes"):
        observations = env.reset()
        total_reward_tug1 = 0
        total_reward_tug2 = 0
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Sample random actions
            actions = {
                'tugboat_1': env.action_space['tugboat_1'].sample(),
                'tugboat_2': env.action_space['tugboat_2'].sample()
            }
            
            # Step environment
            observations, rewards, dones, _ = env.step(actions)
            

            total_reward_tug1 += rewards['tugboat_1']
            total_reward_tug2 += rewards['tugboat_2']
            
            done = dones['__all__']
            step += 1
        
        # Store episode rewards
        episode_rewards_tug1.append(total_reward_tug1)
        episode_rewards_tug2.append(total_reward_tug2)
    
    return episode_rewards_tug1, episode_rewards_tug2



def plot_reward_curve(num_episodes=100, smoothing_window=10) -> None:
    """
    Plot reward curve with moving average.

    Args:
        num_episodes (int): number of episodes to collect rewards
        smoothing_window (int): moving average window

    Returns:
        None
    """
    rewards_tug1, rewards_tug2 = collect_rewards(num_episodes)
    
    # Compute moving averages
    def moving_average(data, window):
        cumsum = np.cumsum(np.insert(data, 0, 0)) 
        return (cumsum[window:] - cumsum[:-window]) / window
    
    # Smooth the rewards
    smooth_rewards_tug1 = moving_average(rewards_tug1, smoothing_window)
    smooth_rewards_tug2 = moving_average(rewards_tug2, smoothing_window)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards_tug1, label='Tugboat 1 Raw Rewards', alpha=0.3, color='blue')
    plt.plot(rewards_tug2, label='Tugboat 2 Raw Rewards', alpha=0.3, color='red')
    plt.title('Raw Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Smoothed rewards
    plt.subplot(1, 2, 2)
    plt.plot(smooth_rewards_tug1, label=f'Tugboat 1 (MA={smoothing_window})', color='blue')
    plt.plot(smooth_rewards_tug2, label=f'Tugboat 2 (MA={smoothing_window})', color='red')
    plt.title(f'Moving Average Rewards (Window={smoothing_window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("maddpg_random.png")
    plt.show()


plot_reward_curve(num_episodes=2000)