import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from mappo_agent import MAPPOAgent
from env2 import CaptureTheFlagEnv
import wandb
wandb.init(project="train_mappo", name="MAPPO Training")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def save_models(agents, directory, episode):
    """Save all agent models."""
    os.makedirs(directory, exist_ok=True)
    for agent_id, agent in agents.items():
        model_path = os.path.join(directory, f"{agent_id}_ep{episode}.pth")
        torch.save(agent.policy.state_dict(), model_path)
    print(f"Models saved for episode {episode}.")


def train_mappo(env, num_episodes=1000, max_steps_per_episode=200, log_dir="logs"):
    """Train agents using MAPPO with logging and saving."""
    agents = {
        agent_id: MAPPOAgent(
            obs_shape=(env.grid_size[0], env.grid_size[1], env.num_agents),
            action_space=env.action_space.n
        ) for agent_id in env.positions.keys()
    }
    for agent in agents.values():
        agent.to(device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{log_dir}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    model_save_dir = os.path.join(log_dir, "models")
    plt.ion()
    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    ax[0].set_title("Agent Movements")
    ax[1].set_title("Episode Rewards")
    ax[1].set_xlabel("Episodes")
    ax[1].set_ylabel("Total Rewards")
    ax[2].set_title("Average Loss")
    ax[2].set_xlabel("Episodes")
    ax[2].set_ylabel("Loss")
    reward_history_team_1 = []
    reward_history_team_2 = []
    loss_history = []

    for episode in range(num_episodes):
        env.reset()
        _, state = env.reset()
        state = {agent_id: np.array(state[agent_id]).flatten() for agent_id in state}
        total_rewards = {"team_1": 0, "team_2": 0}
        episode_losses = []

        for step in range(max_steps_per_episode):
            actions = {}
            log_probs = {}
            for agent_id, agent in agents.items():
                state_tensor = torch.FloatTensor(state[agent_id]).unsqueeze(0).to(device)
                action, log_prob = agent.policy.get_action(state_tensor)
                actions[agent_id] = action.item()
                log_probs[agent_id] = log_prob.detach()

            next_state, rewards, done, scores = env.step(actions)
            next_state = {agent_id: np.array(next_state[agent_id]).flatten() for agent_id in next_state}
            for agent_id, agent in agents.items():
                done_flag = 1 if done["__all__"] else 0
                loss = agent.update(
                    state[agent_id],
                    actions[agent_id],
                    log_probs[agent_id],
                    rewards[agent_id],
                    next_state[agent_id],
                    done_flag,
                    device=device
                )
                if loss is not None:
                    episode_losses.append(loss)
                if agent_id.startswith("team_1"):
                    total_rewards["team_1"] += rewards[agent_id]
                elif agent_id.startswith("team_2"):
                    total_rewards["team_2"] += rewards[agent_id]

            state = next_state
            visualize_movement(env, ax[0], step, episode, scores)
            plt.pause(0.1)

            if done["__all__"]:
                break
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        loss_history.append(avg_loss)
        reward_history_team_1.append(total_rewards["team_1"])
        reward_history_team_2.append(total_rewards["team_2"])
        wandb.log({
            "episode": episode + 1,
            "team_1_rewards": total_rewards["team_1"],
            "team_2_rewards": total_rewards["team_2"],
            "average_loss": avg_loss
        })
        if (episode + 1) % 50 == 0:
            save_models(agents, model_save_dir, episode + 1)

        # Update plots
        visualize_rewards(ax[1], reward_history_team_1, reward_history_team_2, episode + 1)
        visualize_loss(ax[2], loss_history, episode + 1)

        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Team 1 Rewards = {total_rewards['team_1']}, "
              f"Team 2 Rewards = {total_rewards['team_2']}, "
              f"Average Loss = {avg_loss:.4f}")
    save_models(agents, model_save_dir, "final")

    plt.ioff()
    plt.show()
    wandb.finish()


def visualize_loss(ax, loss_history, episode):
    """Update the loss plot in real-time."""
    ax.clear()
    ax.plot(range(1, episode + 1), loss_history, label="Average Loss", color="orange")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    ax.set_title("Episode Loss")
    ax.legend()


def visualize_movement(env, ax, step, episode, scores):
    """Render the current state of the environment in real-time."""
    ax.clear()
    ax.set_xticks(np.arange(0, env.grid_size[1] + 1, 1))
    ax.set_yticks(np.arange(0, env.grid_size[0] + 1, 1))
    ax.grid(True)

    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle((obs[1], obs[0]), 1, 1, color="grey"))
    for agent, pos in env.positions.items():
        color = "cyan" if "team_1" in agent else "green"
        ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, color=color))
        ax.text(pos[1] + 0.5, pos[0] + 0.5, agent[-1], ha="center", va="center", color="black")
    for flag, pos in env.flags.items():
        color = "blue" if "team_1" in flag else "red"
        ax.add_patch(plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.4, color=color))

    ax.set_xlim(0, env.grid_size[1])
    ax.set_ylim(0, env.grid_size[0])
    ax.set_aspect("equal")
    ax.set_title(f"Episode {episode + 1}, Step {step + 1} | Scores: {scores}")


def visualize_rewards(ax, team_1_rewards, team_2_rewards, episode):
    """Update the rewards plot in real-time."""
    ax.clear()
    ax.plot(range(1, episode + 1), team_1_rewards, label="Team 1 Rewards", color="cyan")
    ax.plot(range(1, episode + 1), team_2_rewards, label="Team 2 Rewards", color="green")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Rewards")
    ax.set_title("Episode Rewards")
    ax.legend()


if __name__ == "__main__":
    grid_size = (10, 10)
    team_size = 2
    env = CaptureTheFlagEnv(grid_size=grid_size, team_size=team_size)

    train_mappo(env, num_episodes=1000, max_steps_per_episode=200, log_dir="logs")
