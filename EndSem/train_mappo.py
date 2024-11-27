import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import os
from env_mappo import DenseTetheredBoatsEnv
from mappo import MAPPO


def train_mappo(
    # Training parameters
    num_episodes=10000,
    max_steps_per_episode=100,
    eval_frequency=100,  # Evaluate every N episodes
    # Environment parameters
    grid_size=10,
    tether_length=3,
    # MAPPO parameters
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    epochs_per_update=10,
    batch_size=64,
    # Saving parameters
    save_dir="saved_models",
    experiment_name=None,
):
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories
    model_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(f"runs/tethered_boats_{experiment_name}")

    # Initialize environment
    env = DenseTetheredBoatsEnv(
        grid_size=grid_size,
        tether_length=tether_length,
        step_per_episode=max_steps_per_episode,
    )

    # Initialize MAPPO agent
    mappo = MAPPO(
        env=env,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        epochs=epochs_per_update,
        batch_size=batch_size,
    )

    # Training loop
    best_eval_reward = float("-inf")
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get valid actions
            valid_actions = state["valid_actions"]

            # Sequential action selection
            action1, log_prob1 = mappo.select_action(state, 0, valid_actions[0])
            intermediate_state = env.step_agent(0, action1)
            action2, log_prob2 = mappo.select_action(
                intermediate_state, 1, valid_actions[1]
            )

            # Take step in environment
            next_state, reward, done, _ = env.step([action1, action2])

            # Store transition
            mappo.memory.push(
                state=state,
                actions=[action1, action2],
                reward=reward,
                next_state=next_state,
                valid_actions=valid_actions,
                log_probs=[log_prob1, log_prob2],
                value=mappo.critic(
                    [state["grid"]],
                    [state["pos1"]],
                    [state["pos2"]],
                    torch.tensor([action1]),
                    torch.tensor([action2]),
                ).item(),
                done=done,
            )

            state = next_state
            episode_reward += reward

            # Update if memory is full
            if len(mappo.memory.states) >= mappo.memory.batch_size:
                mappo.update()

        episode_rewards.append(episode_reward)

        # Logging
        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Train/Trash_Remaining", len(env.trash_positions), episode)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Avg Reward (last 10): {avg_reward:.2f}")

        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            eval_rewards = evaluate_agent(env, mappo, num_episodes=5)
            avg_eval_reward = np.mean(eval_rewards)
            writer.add_scalar("Eval/Average_Reward", avg_eval_reward, episode)

            print(f"\nEvaluation at episode {episode + 1}")
            print(f"Average Eval Reward: {avg_eval_reward:.2f}")

            # Save if best model
            if avg_eval_reward > best_eval_reward:
                best_eval_reward = avg_eval_reward
                save_checkpoint(
                    mappo,
                    os.path.join(model_dir, "best_model.pth"),
                    episode,
                    best_eval_reward,
                )

        # Regular checkpoint saving
        if (episode + 1) % 1000 == 0:
            save_checkpoint(
                mappo,
                os.path.join(model_dir, f"checkpoint_{episode+1}.pth"),
                episode,
                avg_eval_reward,
            )

    writer.close()
    return mappo


def evaluate_agent(env, mappo, num_episodes=5, render=False):
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            valid_actions = state["valid_actions"]

            # Get actions from both agents (no exploration)
            with torch.no_grad():
                action1, _ = mappo.select_action(state, 0, valid_actions[0])
                intermediate_state = env.step_agent(0, action1)
                action2, _ = mappo.select_action(
                    intermediate_state, 1, valid_actions[1]
                )

            # Take step in environment
            state, reward, done, _ = env.step([action1, action2])
            episode_reward += reward

            if render:
                env.render()

        rewards.append(episode_reward)

    return rewards


def save_checkpoint(mappo, path, episode, eval_reward):
    torch.save(
        {
            "episode": episode,
            "actor1_state_dict": mappo.actor1.state_dict(),
            "actor2_state_dict": mappo.actor2.state_dict(),
            "critic_state_dict": mappo.critic.state_dict(),
            "actor1_optimizer_state_dict": mappo.actor1_opt.state_dict(),
            "actor2_optimizer_state_dict": mappo.actor2_opt.state_dict(),
            "critic_optimizer_state_dict": mappo.critic_opt.state_dict(),
            "eval_reward": eval_reward,
        },
        path,
    )


def load_checkpoint(mappo, path):
    checkpoint = torch.load(path)
    mappo.actor1.load_state_dict(checkpoint["actor1_state_dict"])
    mappo.actor2.load_state_dict(checkpoint["actor2_state_dict"])
    mappo.critic.load_state_dict(checkpoint["critic_state_dict"])
    mappo.actor1_opt.load_state_dict(checkpoint["actor1_optimizer_state_dict"])
    mappo.actor2_opt.load_state_dict(checkpoint["actor2_optimizer_state_dict"])
    mappo.critic_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    return checkpoint["episode"], checkpoint["eval_reward"]


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Training configuration
    config = {
        "num_episodes": 10000,
        "max_steps_per_episode": 100,
        "eval_frequency": 100,
        "grid_size": 10,
        "tether_length": 3,
        "lr_actor": 3e-4,
        "lr_critic": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "epochs_per_update": 10,
        "batch_size": 64,
    }

    # Start training
    print("Starting training with configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    mappo = train_mappo(**config)

    # Test trained agent
    print("\nTesting trained agent...")
    eval_rewards = evaluate_agent(mappo.env, mappo, num_episodes=5, render=True)
    print(f"Average test reward: {np.mean(eval_rewards):.2f}")
