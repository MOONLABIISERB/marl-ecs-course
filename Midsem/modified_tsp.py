"""Environment for Modified Travelling Salesman Problem."""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from numpy import typing as npt

from modTSP import ModTSP

from agent import Agent

import wandb
import torch
from modTSP import ModTSP
from agent import Agent, DQN

from auxiliary import run_inference

import argparse


def get_epsilon(episode, min_epsilon=0.01, max_epsilon=0.7, decay_rate=0.0005):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


def main() -> None:
    """Main function."""
    num_targets = 10
    num_episodes = 10**4
    max_steps_per_episode = num_targets

    env = ModTSP(num_targets)
    obs, _ = env.reset()
    ep_rets = []

    agent = Agent(10**4, env.observation_space.shape[0], env.action_space.n, 0.99)
    batchSize = 256

    for ep in range(num_episodes):
        ret = 0
        done = False
        losses = []
        obs, _ = env.reset()
        soft_upper_max_profit = env.calculate_soft_upper_max_profit()
        # print(f"Episode {ep}, Soft Upper Max Profit: {soft_upper_max_profit:.2f}")

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

        avg_loss = np.sum(losses) if losses else 1000
        if losses:
            avg_loss = np.mean(losses)
            wandb.log(
                {
                    "avg_loss": avg_loss,
                    "episode_reward": ret,
                    "Profit": np.sum(episode_profits),
                    "Average Distance": np.mean(episode_distances),
                    "soft_upper_max_profit": soft_upper_max_profit,
                    "profit_ratio": ret / soft_upper_max_profit if soft_upper_max_profit > 0 else 0,
                    "Total Distance": total_distance,
                }
            )

        if ep % 100 == 0:
            agent.update_knowledge()

        if ep % 100 == 0:
            print(
                f"Episode {ep},  Reward: {ret:.2f}, "
                f"Profit: {np.sum(episode_profits):.2f}, "
                f"Distance: {np.sum(episode_distances):.2f}",
                f"Loss: {avg_loss:.2f}",
                f"Initial Profits: {env.initial_profits}",
                # f"Current profits {info["current_profits"]}",
            )
        ep_rets.append(ret)
        env.episodes += 1

    print(np.mean(ep_rets))
    agent.save_model("model.pth")
    wandb.finish()


if __name__ == "__main__":
    # add argparse for train or test
    parser = argparse.ArgumentParser(description="Train or test the model.")
    parser.add_argument("--mode", choices=["train", "test"], help="Mode to run: train or test")
    args = parser.parse_args()

    if args.mode == "test":
        run_inference("model.pth")
        exit()

    elif args.mode != "train":
        print("Invalid mode. Please choose either 'train' or 'test'.")
        exit()

    # Initialize wandb
    wandb.init(project="marl", name="mtsp_final")

    main()

    wandb.finish()

    # # Run inference after training
    # run_inference("model.pth")
