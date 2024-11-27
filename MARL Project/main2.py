import gymnasium as gym
import rware
import time
import numpy as np
from actor_critic import Agent
from rware import RewardType
import matplotlib.pyplot as plt

# Plotting learning curve
def plot_learning_curve(x, scores, filename):
    plt.figure()
    plt.plot(x, scores, label='Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    layout = """
......
..xx..
..xx..
..xx..
......
..gg..
"""

    env = gym.make("rware-tiny-2ag-v2", layout=layout, reward_type=RewardType.TWO_STAGE)
    agents = [Agent(alpha=1e-5, n_actions=env.action_space[0].n, name=f"agent_{i+1}") for i in range(env.n_agents)]
    n_games = 500
    n_steps = 500
    filename = 'warehouse.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        for agent in agents:
            agent.load_models()

    for i in range(n_games):
        observations, info = env.reset()
        done1 = [False] * env.n_agents
        score = [0] * env.n_agents
        stuck_steps = [0] * env.n_agents
        for j in range(n_steps):
            actions = [agents[k].choose_action(observations[k]) for k in range(env.n_agents)]
            next_observations, rewards, done, truncated, info = env.step(tuple(actions))
            for k in range(env.n_agents):
                #rewards[k] = int(rewards[k])
                score[k] += rewards[k]
                if not load_checkpoint:
                    agents[k].learn(observations[k], rewards[k], next_observations[k], done)
            observations = next_observations
            env.render()
        score_history.append(sum(score))
        avg_score = np.mean(score_history[-1:])
        if (avg_score >= best_score and best_score != 0) or (avg_score>best_score and best_score ==0):
            best_score = avg_score
            if not load_checkpoint:
                for agent in agents:
                    agent.save_models()
        print('episode ', i, 'score %.1f' % sum(score), 'avg_score %.1f' % avg_score)

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
    env.close()

# %%
env.close()

# %%
