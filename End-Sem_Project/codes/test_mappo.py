import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from env2 import CaptureTheFlagEnv
from mappo_agent import MAPPOAgent


def load_models(agents, model_dir):
    """Load trained models for agents."""
    for agent_id, agent in agents.items():
        model_path = os.path.join(model_dir, f"{agent_id}_epfinal.pth")
        if os.path.exists(model_path):
            agent.policy.load_state_dict(torch.load(model_path))
            print(f"Loaded model for {agent_id} from {model_path}")
        else:
            print(f"Model for {agent_id} not found at {model_path}")


def test_agents(env, agents, num_episodes=100, max_steps_per_episode=200, log_dir="test_logs"):
    """Test trained MAPPO agents with real-time visualization and scoring logic."""
    results = {"team_1_wins": 0, "team_2_wins": 0}
    scores_history = []
    movements = []

    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].set_title("Agent Movements During Testing")
    ax[1].set_title("Episode Scores")
    ax[1].set_xlabel("Episodes")
    ax[1].set_ylabel("Total Scores")
    scores_team_1 = []
    scores_team_2 = []

    for episode in range(num_episodes):
        env.reset()
        _, state = env.reset()
        state = {agent_id: np.array(state[agent_id]).flatten() for agent_id in state}
        episode_scores = {"team_1": 0, "team_2": 0}
        episode_movements = []

        for step in range(max_steps_per_episode):
            actions = {}
            for agent_id, agent in agents.items():
                state_tensor = torch.FloatTensor(state[agent_id]).unsqueeze(0)
                action, _ = agent.policy.get_action(state_tensor)
                actions[agent_id] = action.item()

            next_state, rewards, done, scores = env.step(actions)
            next_state = {agent_id: np.array(next_state[agent_id]).flatten() for agent_id in next_state}
            episode_movements.append({agent_id: env.positions[agent_id] for agent_id in agents.keys()})
            for agent_id, reward in rewards.items():
                if agent_id.startswith("team_1"):
                    episode_scores["team_1"] += reward
                elif agent_id.startswith("team_2"):
                    episode_scores["team_2"] += reward

            if done["__all__"]:
                if scores["team_1"] > scores["team_2"]:
                    results["team_1_wins"] += 1
                    print(f"Team 1 wins Episode {episode + 1}!")
                elif scores["team_2"] > scores["team_1"]:
                    results["team_2_wins"] += 1
                    print(f"Team 2 wins Episode {episode + 1}!")
                break

            state = next_state
        scores_history.append(episode_scores)
        movements.append(episode_movements)
        scores_team_1.append(episode_scores["team_1"])
        scores_team_2.append(episode_scores["team_2"])
        visualize_movement(env, ax[0], step, episode, scores)
        visualize_scores(ax[1], scores_team_1, scores_team_2, episode + 1)
        plt.pause(0.1)

    plt.ioff()
    plt.show()

    print("\nFinal Testing Results:")
    print(f"Team 1 Wins: {results['team_1_wins']}")
    print(f"Team 2 Wins: {results['team_2_wins']}")
    os.makedirs(log_dir, exist_ok=True)
    torch.save(scores_history, os.path.join(log_dir, "scores.pt"))
    torch.save(movements, os.path.join(log_dir, "movements.pt"))
    print(f"Test results saved in {log_dir}")


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


def visualize_scores(ax, scores_team_1, scores_team_2, episode):
    """Update the scores plot in real-time."""
    ax.clear()
    ax.plot(range(1, episode + 1), scores_team_1, label="Team 1 Scores", color="cyan")
    ax.plot(range(1, episode + 1), scores_team_2, label="Team 2 Scores", color="green")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Scores")
    ax.set_title("Episode Scores")
    ax.legend()


if __name__ == "__main__":
    grid_size = (10, 10)
    team_size = 2
    model_dir = "/home/elliot/Desktop/Shirish/PhD/Courses/MARL-ECS/Project/codes/logs/20241124-174906/models"  # Update this with the path to your saved models
    log_dir = "test_logs_mappo"
    env = CaptureTheFlagEnv(grid_size=grid_size, team_size=team_size)
    agents = {
        agent_id: MAPPOAgent(
            obs_shape=(env.grid_size[0], env.grid_size[1], env.num_agents),
            action_space=env.action_space.n
        ) for agent_id in env.positions.keys()
    }

    # Load trained models
    load_models(agents, model_dir)

    # Test agents
    test_agents(env, agents, num_episodes=100, max_steps_per_episode=200, log_dir=log_dir)
