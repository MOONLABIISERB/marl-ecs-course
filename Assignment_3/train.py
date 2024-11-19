from environment import MultiAgentEnv
import pickle
from agent import QLearningAgent

def marl_learning_rollout(env, agents, num_rollouts=100, max_steps = 10000):
    total_rewards = {i: 0 for i in range(len(agents))}
    rewards = {i: [] for i in range(len(agents))}
    q_table = {}
    for episode in range(num_rollouts):
        obs = env.reset()
        done = False
        episode_reward = {i: 0 for i in range(len(agents))}
        step = 0
        while not done and step < max_steps:
            actions = {}
            next_actions = {}
            for agent_id, agent in agents.items():
                actions[agent_id], q_table = agent.choose_action(obs, q_table)

            
            next_obs, reward, done, _ = env.step(actions)
            
            for agent_id, agent in agents.items():
                next_actions[agent_id], q_table = agent.choose_action(next_obs, q_table)
                episode_reward[agent_id] += reward[agent_id]
                q_table = agent.learn(obs, actions[agent_id], reward[agent_id], next_obs, done, q_table)
    
            obs = next_obs
            step += 1
        
        if episode % (num_rollouts / 1000) == 0:
            print(f'reward at episode: {episode} is {episode_reward}')

        for agent_id in agents.keys():
            rewards[agent_id].append(episode_reward)
            total_rewards[agent_id] += episode_reward[agent_id]

    return rewards, {key: value / num_rollouts for key, value in total_rewards.items()}, q_table

def save_q_tables(q_table, filename_prefix='q_table_agent'):
    filename = f"Assignment_3/{filename_prefix}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)
        print(f"q_table saved to {filename}")

if __name__ == "__main__":

    goals = {
        0: (1,1),
        1: (8,1),
        2: (1,8),
        3: (8,8)
    }

    obstacles=[
        (0, 4),
        (1, 4),
        (2, 4),
        (2, 5),
        (4, 0),
        (4, 1),
        (4, 2),
        (5, 2),
        (4, 9),
        (4, 8),
        (4, 7),
        (5, 7),
        (9, 5),
        (8, 5),
        (7, 5),
        (7, 4)
    ]

    max_steps = 100_000
    num_agents = 4
    num_rollouts = 1_000_000

    epsilon = 0.1
    alpha = 0.2
    gamma = 0.9

    env = MultiAgentEnv(grid_size=(10, 10), goals=goals, obstacles=obstacles, num_agents=num_agents)

    agents = {i: QLearningAgent(agent_id=i, action_space=env.action_spaces[i], epsilon=epsilon, alpha=alpha, gamma=gamma) for i in range(num_agents)}

    rewards, average_rewards, q_table = marl_learning_rollout(env=env, agents=agents, num_rollouts=num_rollouts, max_steps=max_steps)

    print(f"Average rewards for each agent over {num_rollouts} rollouts:{average_rewards}")

    save_q_tables(q_table)
