
# from rware.warehouse import Warehouse, RewardType
# import sys
# import os
# import numpy as np
# import gymnasium as gym
# from dql import DQNAgent
# import matplotlib.pyplot as plt

# def plot_learning_curve(x, scores, filename):
#     plt.figure()
#     plt.plot(x, scores, label='Mean Score (last 10 episodes)')
#     plt.xlabel('Episode Group (10 episodes)')
#     plt.ylabel('Average Score')
#     plt.title('Learning Curve (Grouped by 10 Episodes)')
#     plt.legend()
#     plt.savefig(filename)
#     plt.show()

# if __name__ == '__main__':
#     layout = """
#     ......  
#     ..xx..
#     ..xx..
#     ..xx..
#     ......  
#     ..gg..
#     """
#     env = gym.make("rware-tiny-2ag-v2", layout=layout, reward_type=RewardType.TWO_STAGE)
#     n_agents = 2
#     input_dims = env.observation_space[0].shape
#     n_actions = env.action_space[0].n

#     agents = [DQNAgent(gamma=0.95, epsilon=1.0, batch_size=64, n_actions=n_actions,
#                        input_dims=input_dims, lr=0.001) for _ in range(n_agents)]

#     n_games = 80
#     max_steps = 500
#     episode_scores = []  # Store the total scores of all agents for each episode
#     mean_scores_10 = []  # Store the mean scores for every 10 episodes

#     for i in range(n_games):
#         observations = env.reset()[0]
#         total_score = 0  # Total score for all agents in this episode

#         for step in range(max_steps):
#             actions = [agent.choose_action(observations[agent_id]) for agent_id, agent in enumerate(agents)]
#             observations_, rewards, dones, _, _ = env.step(tuple(actions))
#             env.render()
#             rewards = list(rewards)

#             for agent_id, agent in enumerate(agents):
#                 agent.store_transition(observations[agent_id], actions[agent_id],
#                                        rewards[agent_id], observations_[agent_id], False)
#                 if rewards[agent_id] != -0.4:
#                     total_score += rewards[agent_id]
#                 agent.learn()

#             observations = observations_

#         episode_scores.append(total_score)

#         # After every 10 episodes, calculate and store the mean score for the last 10 episodes
#         if (i + 1) % 5 == 0:
#             mean_scores_10.append(np.mean(episode_scores[-5:]))

#         print(f"Episode {i+1}/{n_games}: Total Score: {total_score}")

#     # Plot the mean scores for every 10 episodes
#     x = [i + 1 for i in range(len(mean_scores_10))]
#     filename = 'rware_grouped_mean_scores.png'
#     plot_learning_curve(x, mean_scores_10, filename)

#     env.close()



# # %%
# import sys
# import os
# from gymnasium.envs.registration import register
# import numpy as np
# import time
# import gymnasium as gym
# from dql import DQNAgent
# import matplotlib.pyplot as plt
# # Add the rware directory to Python path
# # rware_path = os.path.expanduser("~/Desktop/rware")
# # sys.path.append(rware_path)

# def plot_learning_curve(x, scores, filename):
#     plt.figure()
#     plt.plot(x, scores, label='Score')
#     plt.xlabel('Episode')
#     plt.ylabel('Score')
#     plt.title('Learning Curve')
#     plt.legend()
#     plt.savefig(filename)
#     plt.show()
# # # Import Warehouse and RewardType
# from rware.warehouse import Warehouse, RewardType

# # # Register the environment
# # register(
# #     id="Warehouse-v0",  # Unique ID for this environment
# #     entry_point="rware.warehouse:Warehouse",
# #     kwargs={
# #         "shelf_columns": 3,
# #         "column_height": 3,
# #         "shelf_rows": 2,
# #         "n_agents": 1,
# #         "msg_bits": 0,
# #         "sensor_range": 2,
# #         "request_queue_size": 3,
# #         "max_inactivity_steps": None,
# #         "max_steps": 500,
# #         "reward_type": RewardType.TWO_STAGE,
# #         "layout": None,
# #     },
# # )

# # Use the registered environment

# import gymnasium as gym
# import rware
# #%%
# if __name__ == '__main__':
#     layout = """
#     ......
#     ..xx..
#     ..xx..
#     ..xx..
#     ......
#     ..gg..
#     """
#     # layout = """
#     # ........
#     # ........
#     # ...xx...
#     # ...xx...
#     # ...xx...
#     # ........
#     # ........
#     # ...gg...
#     # """
#     env = gym.make("rware-tiny-2ag-v2", layout=layout, reward_type=RewardType.TWO_STAGE)
#     n_agents = 2
#     input_dims = env.observation_space[0].shape
#     n_actions = env.action_space[0].n

#     agents = [DQNAgent(gamma=0.95, epsilon=1.0, batch_size=64, n_actions=n_actions,
#                        input_dims=input_dims, lr=0.001) for _ in range(n_agents)]

#     n_games = 80
#     max_steps = 500
#     scores = [[] for _ in range(n_agents)]
#     eps_history = [[] for _ in range(n_agents)]

#     for i in range(n_games):
#         observations = env.reset()[0]
#         episode_scores = [0] * n_agents

#         for step in range(max_steps):
#             actions = [agent.choose_action(observations[agent_id]) for agent_id, agent in enumerate(agents)]
#             observations_, rewards, dones, _, _ = env.step(tuple(actions))
#             env.render()
#             rewards = list(rewards)

#             for agent_id, agent in enumerate(agents):
#                 agent.store_transition(observations[agent_id], actions[agent_id],
#                                        rewards[agent_id], observations_[agent_id], False)
#                 if rewards[agent_id] != -0.4:
#                     episode_scores[agent_id] += rewards[agent_id]
#                 agent.learn()

#             observations = observations_

#         for agent_id in range(n_agents):
#             scores[agent_id].append(episode_scores[agent_id])
#             eps_history[agent_id].append(agents[agent_id].epsilon)

#         avg_scores = [np.mean(scores[agent_id][-100:]) for agent_id in range(n_agents)]
#         print(f"Episode {i+1}/{n_games}: ", ", ".join(
#             [f"Agent {agent_id} score: {episode_scores[agent_id]}" for agent_id in range(n_agents)]))

#     for agent_id in range(n_agents):
#         x = [i + 1 for i in range(n_games)]
#         filename = f'rware_agent_{agent_id}.png'
#         plot_learning_curve(x, scores[agent_id], filename)

#     env.close()
# # %%
# env.close()

# # %%
# env.close()
# # %%

# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import rware
from rware.warehouse import RewardType
from dql import DQNAgent

def plot_learning_curve(x, scores, filename):
    plt.figure()
    plt.plot(x, scores, label='Average Score')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(filename)
    plt.show()

# %%
if __name__ == '__main__':
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

    env = gym.make("rware-tiny-2ag-v2", layout=layout, reward_type=RewardType.TWO_STAGE)
    n_agents = 2
    input_dims = env.observation_space[0].shape
    n_actions = env.action_space[0].n

    agents = [DQNAgent(gamma=0.95, epsilon=1.0, batch_size=64, n_actions=n_actions,
                       input_dims=input_dims, lr=0.001) for _ in range(n_agents)]

    n_games = 2600
    max_steps = 800
    scores = []
    avg_scores = []

    for i in range(n_games):
        observations = env.reset()[0]
        episode_scores = [0] * n_agents

        for step in range(max_steps):
            actions = [agent.choose_action(observations[agent_id]) for agent_id, agent in enumerate(agents)]
            observations_, rewards, dones, _, _ = env.step(tuple(actions))
            env.render()
            rewards = list(rewards)

            for agent_id, agent in enumerate(agents):
                agent.store_transition(observations[agent_id], actions[agent_id],
                                       rewards[agent_id], observations_[agent_id], False)
                if round(rewards[agent_id], 1) != -0.6 and round(rewards[agent_id], 1) != -0.3:
                    episode_scores[agent_id] += rewards[agent_id]
                agent.learn()

            observations = observations_

        # Sum up the scores for all agents for the current episode
        total_episode_score = sum(episode_scores)
        scores.append(total_episode_score)

        # Calculate and store the average score for the last 5 episodes
        if (i + 1) % 5 == 0:
            avg_score = np.mean(scores[-5:])
            avg_scores.append(avg_score)
            print(f"Episode {i+1}: Average score of last 5 episodes: {avg_score}")

    # Plot the averaged scores every 5 episodes
    x = [5 * (i + 1) for i in range(len(avg_scores))]
    filename = 'rware_avg_scores.png'
    plot_learning_curve(x, avg_scores, filename)

    env.close()
    # %%
    import torch  as T
    def save_models(agents, save_dir='saved_models'):
        """Save all agents' models to disk"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for i, agent in enumerate(agents):
            # Save both Q_eval and Q_target networks
            T.save(agent.Q_eval.state_dict(), f'{save_dir}/agent_{i}_Q_eval.pth')
            T.save(agent.Q_target.state_dict(), f'{save_dir}/agent_{i}_Q_target.pth')

    save_models(agents)        

# %%
input_dims = env.observation_space[0].shape
print(input_dims)
# %%
