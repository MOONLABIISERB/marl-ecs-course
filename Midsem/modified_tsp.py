"""Environment for Modified Travelling Salesman Problem."""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from numpy import typing as npt

from modTSP import ModTSP

from agent import Agent

import wandb


def get_epsilon(episode, min_epsilon=0.01, max_epsilon=0.7, decay_rate=0.0005):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


def main() -> None:
    """Main function."""
    num_targets = 10
    num_episodes = 10**5
    max_steps_per_episode = num_targets

    env = ModTSP(num_targets)
    obs, _ = env.reset()
    ep_rets = []

    agent = Agent(10**5, env.observation_space.shape[0], env.action_space.n, 0.99)
    batchSize = 10**2

    for ep in range(num_episodes):
        ret = 0
        done = False
        losses = []
        obs, _ = env.reset()
        epsilon = get_epsilon(ep)
        n = 0

        episode_rewards = []
        episode_profits = []
        episode_distances = []
        total_distance = 0

        for _ in range(max_steps_per_episode):
            # action = env.action_space.sample()  # You need to replace this with your algorithm that predicts the action.
            action = agent.action(obs, exploration_prob=epsilon)
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += reward

            total_distance += info["distance_travelled"]

            agent.update_memory(obs, action, reward, obs_, done)
            obs = obs_

            # Learn and track loss
            if len(agent.memory) >= batchSize:
                loss = agent.learn_from_memory(batch_size=batchSize)
                if loss is not None:
                    losses.append(loss)
            if done:
                break

            episode_rewards.append(reward)
            episode_profits.append(env.current_profits[info["profit"]])
            episode_distances.append(info["distance_travelled"])

        avg_loss = np.mean(losses) if losses else 1000
        if losses:
            avg_loss = np.mean(losses)
            wandb.log({"avg_loss": avg_loss, "episode_reward": ret, "Average Distance": np.sum(episode_distances) / len(episode_distances)})

        if ep % 30 == 0:
            agent.update_knowledge()

        if ep % 100 == 0:
            print(
                f"Episode {ep}, Avg Reward: {np.mean(episode_rewards):.2f}, "
                f"Total Reward: {np.sum(episode_rewards):.2f}, "
                f"Average Profit: {np.mean(episode_profits):.2f}, "
                f"Average Distance: {np.mean(episode_distances):.2f}",
                f"Current profits {info["current_profits"]}",
            )
        ep_rets.append(ret)

    print(np.mean(ep_rets))
    wandb.finish()


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="marl", name="mtsp_1")

    main()
