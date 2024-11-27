import wandb
from env import TetheredBoatsEnv
from agent import CentralizedAgent
import numpy as np
import torch
import argparse


def train(args):
    # Initialize wandb
    wandb.init(
        project="tethered-boats",
        config={
            "grid_size": args.grid_size,
            "n_boats": 2,
            "tether_length": args.tether_length,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "target_update": args.target_update,
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
        },
    )

    # Create environment and agent
    env = TetheredBoatsEnv(
        grid_size=args.grid_size,
        tether_length=args.tether_length,
        step_per_episode=args.max_steps,
        num_episode=args.n_episodes,
    )

    agent = CentralizedAgent(
        env,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        # epsilon_start=args.epsilon_start,
        # epsilon_end=args.epsilon_end,
        # epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
    )

    # Training loop
    for episode in range(args.n_episodes):
        state = env.reset()
        episode_reward = 0
        trash_collected = 0
        initial_trash = len(env.trash_positions)

        for step in range(args.max_steps):
            # Select action
            action_idx = agent.select_action(state, env.boat_positions)
            actions = agent.decode_action(action_idx)

            boat_pos = env.boat_positions.copy()

            # Take action
            next_state, reward, done, _ = env.step(actions)

            next_boat_pos = env.boat_positions.copy()

            # Store experience
            # Ensure states are properly shaped for the CNN
            agent.step(
                state.copy(),
                action_idx,
                reward,
                next_state.copy(),
                done,
                boat_pos,
                next_boat_pos,
            )

            # Update metrics
            episode_reward += reward
            trash_collected = initial_trash - len(env.trash_positions)

            # Render if needed
            if args.render and episode % args.render_freq == 0:
                env.render()

            # Log step metrics
            wandb.log(
                {
                    "step_reward": reward,
                    # "epsilon": agent.epsilon,
                    "trash_remaining": len(env.trash_positions),
                }
            )

            if done:
                break

            state = next_state

        # Log episode metrics
        wandb.log(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_length": step + 1,
                "trash_collected": trash_collected,
                "completion_rate": trash_collected / initial_trash,
            }
        )

        # Print progress
        print(
            f"Episode {episode}/{args.n_episodes} - Reward: {episode_reward:.2f} - "
            f"Trash Collected: {trash_collected}/{initial_trash}"
        )

        # Save model periodically
        if episode % args.save_freq == 0:
            torch.save(
                agent.qnetwork_local.state_dict(),
                f"models/tethered_boats_model_ep{episode}.pth",
            )
            wandb.save(f"models/tethered_boats_model_ep{episode}.pth")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tethered Boats Agent")

    # Environment parameters
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--tether_length", type=int, default=3)

    # Training parameters
    parser.add_argument("--n_episodes", type=int, default=100000)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_update", type=int, default=10)

    # Logging and saving
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=100)

    args = parser.parse_args()
    train(args)
