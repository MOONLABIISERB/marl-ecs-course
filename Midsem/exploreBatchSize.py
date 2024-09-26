import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from collections import namedtuple, deque
import gymnasium as gym
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

# ... (ReplayMemory, DQN, and Agent classes remain the same)
from agent import Agent


def train_dqn(batch_size, num_episodes=1000):
    wandb.init(project="dqn-cartpole-batch-size", name=f"DQN-CartPole-v1-batch-{batch_size}", config={"batch_size": batch_size})

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(memory_capacity=10000, state_dim=state_dim, n_actions=n_actions, gamma=0.99)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_losses = []

        while not done:
            action = agent.action(state, exploration_prob=0.1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update_memory(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(agent.memory) >= batch_size:
                loss = agent.learn_from_memory(batch_size=batch_size)
                if loss is not None:
                    episode_losses.append(loss)

        if episode_losses:
            avg_loss = np.mean(episode_losses)
            wandb.log({"episode": episode, "avg_loss": avg_loss, "episode_reward": episode_reward})
        else:
            avg_loss = 0

        if episode % 10 == 0:
            agent.update_knowledge()
            print(f"Episode {episode}, Avg Loss: {avg_loss:.4f}, Reward: {episode_reward}")

    wandb.finish()
    return np.mean([wandb.run.summary.get("episode_reward", 0) for _ in range(10)])  # Return average of last 10 episodes


def find_optimal_batch_size(batch_sizes=[32, 64, 128, 256]):
    results = []
    for batch_size in batch_sizes:
        avg_reward = train_dqn(batch_size)
        results.append((batch_size, avg_reward))
        print(f"Batch size {batch_size}: Average reward of last 10 episodes = {avg_reward}")

    optimal_batch_size = max(results, key=lambda x: x[1])[0]
    print(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


if __name__ == "__main__":
    optimal_batch_size = find_optimal_batch_size()

    # Final run with optimal batch size
    print(f"Running final training with optimal batch size: {optimal_batch_size}")
    train_dqn(optimal_batch_size, num_episodes=2000)  # Longer final training
