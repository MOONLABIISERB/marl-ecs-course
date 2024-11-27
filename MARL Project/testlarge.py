import gymnasium as gym
import rware
from rware.warehouse import RewardType
import torch as T
from dql import DQNAgent, DeepQNetwork
import numpy as np
import time

def load_agents(n_agents, save_dir='saved_models'):
    """Load trained agents from disk"""
    agents = []
    
    for i in range(n_agents):
        
        # Create agent with loaded parameters
        agent = DQNAgent(
            gamma=0.95,
            epsilon=0.37,  # Use minimum epsilon for testing
            lr=0.001,
            input_dims=(71,),
            batch_size=64,
            n_actions=5,
            max_mem_size=100000
        )
        
        # Load network weights
        agent.Q_eval.load_state_dict(T.load(f'{save_dir}/agentlarge_{i}_Q_eval.pth'))
        
        agents.append(agent)
    
    return agents

def test_agents(n_episodes=10, max_steps=1000):
    # layout = """
    # ......
    # ..xx..
    # ..xx..
    # ..xx..
    # ......
    # ..gg..
    # """

    layout = """
    ......
    ..xx..
    ..xx..
    ..xx..
    ..xx..
    ..xx..
    ..xx..
    ..xx..
    ......
    ..gg..
    """
    env = gym.make("rware-tiny-2ag-v2",layout=layout, reward_type=RewardType.TWO_STAGE)
    n_agents = 2
    
    # Load trained agents
    agents = load_agents(n_agents)
    
    for episode in range(n_episodes):
        observations = env.reset()[0]
        episode_scores = [0] * n_agents
        
        for step in range(max_steps):
            # Get actions from trained agents
            actions = [agent.choose_action(observations[agent_id]) for agent_id, agent in enumerate(agents)]
            observations_, rewards, dones, _, _ = env.step(tuple(actions))
            env.render()
            
            rewards = list(rewards)
            
            # Update episode scores
            for agent_id in range(n_agents):
                if round(rewards[agent_id], 1) != -0.6 and round(rewards[agent_id], 1) != -0.3:
                    episode_scores[agent_id] += rewards[agent_id]
            
            observations = observations_
            time.sleep(0.002)
        
        total_score = sum(episode_scores)
        print(f"Episode {episode + 1}: Total Score = {total_score}")
    
    env.close()

if __name__ == '__main__':
    test_agents()

# %%

# %%
