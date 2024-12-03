
import os
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from sim import Env, ReplayMemory
from sim.agents.agents import AgentDQN
from utils import Config, Metrics, train, test

config = Config('config/')

if not config.save_build:
    plt.ion()
else:
    plt.switch_backend('agg')


device_type = "cuda" if torch.cuda.is_available() and config.learning.cuda else "cpu"
device = torch.device(device_type)

print("Using", device_type)

# Set up save folder structure
if config.save_build:
    name = datetime.today().strftime('%Y-%m-%d %H:%M:%S') if not config.build_name else config.build_name
    root_path = os.path.abspath(config.learning.save_folder + '/' + name)
    model_path = os.path.join(root_path, "models")
    path_figure = os.path.join(root_path, "figs")
    os.makedirs(model_path)
    os.makedirs(path_figure)
    shutil.copytree(os.path.abspath('config/'), os.path.join(root_path, 'config'))

number_agents = config.agents.number_predators + config.agents.number_preys
# Define the agents
agents = [AgentDQN("predator", "predator-{}".format(k), device, config.agents)
          for k in range(config.agents.number_predators)]
agents += [AgentDQN("prey", "prey-{}".format(k), device, config.agents)
           for k in range(config.agents.number_preys)]

metrics = []
collision_metric = Metrics()
memory = ReplayMemory(config.replay_memory.size)

# Define the metrics for all agents
for agent in agents:
    metrics.append(Metrics())

    # If we have to load the pretrained model
    if config.learning.use_model:
        path = os.path.abspath(os.path.join(config.learning.model_path, agent.id + ".pth"))
        agent.load(path)

env = Env(config.env, config)

# Add agents to the environment
for agent in agents:
    env.add_agent(agent, position=None)

# Set up the figures for visualization
fig_board = plt.figure(0, figsize=(10, 10))
if config.env.world_3D:
    ax_board = fig_board.gca(projection="3d")
else:
    ax_board = fig_board.gca()

fig_losses_returns, (ax_losses, ax_returns, ax_collisions) = plt.subplots(3, 1, figsize=(20, 10))

plt.show()

start = time.time()
path_figure_episode = None
action_dim = 7 if config.env.world_3D else 5

progress_bar = None
for episode in range(config.learning.n_episodes):
    if not episode % config.learning.plot_episodes_every:
        if progress_bar is not None:
            progress_bar.close()
        progress_bar = tqdm(total=config.learning.plot_episodes_every)

    # Test step
    if not episode % config.learning.test_every:
        for test_episode in range(config.learning.n_episode_in_test):
            test(env, agents, collision_metric, metrics, config)

    # Plot step
    if not episode % config.learning.plot_episodes_every or not episode % config.learning.save_episodes_every:
        all_states, all_rewards, all_types = test(env, agents, collision_metric, metrics, config)

        # Make path for episode images
        if not episode % config.learning.save_episodes_every and config.save_build:
            path_figure_episode = os.path.join(path_figure, "episode-{}".format(episode))
            os.mkdir(path_figure_episode)

        # Plot last test episode
        for k, (states, rewards, types) in enumerate(zip(all_states, all_rewards, all_types)):
            # Plot environment
            ax_board.cla()
            env.plot(states, types, rewards, ax_board)
            plt.draw()
            if not episode % config.learning.save_episodes_every and config.save_build:
                fig_board.savefig(os.path.join(path_figure_episode, "frame-{}.jpg".format(k)))
                fig_losses_returns.savefig(os.path.join(path_figure, "losses_returns_{}.eps".format(episode)), dpi=1000, format="eps")
            if not episode % config.learning.plot_episodes_every:
                plt.pause(0.001)

    all_states, all_next_states, all_rewards, all_actions, _ = train(env, agents, memory,
                                                                     metrics, action_dim, config)

    # Plot learning curves
    if not episode % config.learning.plot_curves_every:
        print("Episode", episode)
        print("Time :", time.time() - start)
        ax_losses.cla()
        ax_returns.cla()
        ax_collisions.cla()
        for k in range(len(agents)):
            metrics[k].compute_averages()

            metrics[k].plot_losses(episode, ax_losses, legend=agents[k].id)
            metrics[k].plot_returns(episode, ax_returns, legend=agents[k].id)
            ax_losses.set_title("Losses")
            ax_losses.legend()
            ax_returns.set_title("Returns")
            ax_returns.legend()
        collision_metric.compute_averages()
        collision_metric.plot_collision_counts(episode, ax_collisions)
        ax_collisions.set_title("Number of collisions")

        plt.draw()
        plt.pause(0.0001)

        # Save the plots after every plot_curves_every episodes
        if config.save_build:
            fig_losses_returns.savefig(os.path.join(path_figure, "losses_returns_{}.png".format(episode)))

    # Save models after every episode
    if config.save_build:
        for agent in agents:
            path = os.path.join(model_path, agent.id + ".pth")
            agent.save(path)

    progress_bar.update(1)
progress_bar.close()
