import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from env import CaptureTheFlagEnv
from mappo_agent import MAPPOAgent

class QLearningAgent:
    """Independent Q-Learning Agent for testing."""
    def __init__(self, num_actions=5):
        self.q_table = {}
        self.num_actions = num_actions

    def load_q_table(self, q_table):
        self.q_table = q_table

    def choose_action(self, state):
        state_key = tuple(state.flatten())
        if state_key in self.q_table:
            return np.argmax(self.q_table[state_key])
        return np.random.choice(range(self.num_actions))

def load_q_tables(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_mappo_models(agents, model_dir):
    for agent_id, agent in agents.items():
        model_path = os.path.join(model_dir, f"{agent_id}_epfinal.pth")
        if os.path.exists(model_path):
            agent.policy.load_state_dict(torch.load(model_path))
            print(f"Loaded model for {agent_id} from {model_path}")
        else:
            print(f"Model for {agent_id} not found at {model_path}")

def test_agents(env, agents, num_episodes=100, max_steps_per_episode=200, algorithm="IQL", log_dir="test_logs"):
    results = {"team_1_wins": 0, "team_2_wins": 0, "draws": 0}
    scores_history = []
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].set_title("Agent Movements During Testing")
    ax[1].set_title("Episode Scores")
    ax[1].set_xlabel("Episodes")
    ax[1].set_ylabel("Total Scores")
    scores_team_1 = []
    scores_team_2 = []
    win_status = []  # Track win status per episode

    for episode in range(num_episodes):
        _, state = env.reset()
        state = {agent_id: np.array(state[agent_id]).flatten() for agent_id in state}
        episode_scores = {"team_1": 0, "team_2": 0}

        for step in range(max_steps_per_episode):
            actions = {}
            for agent_id, agent in agents.items():
                if algorithm == "MAPPO":
                    state_tensor = torch.FloatTensor(state[agent_id]).unsqueeze(0)
                    action, _ = agent.policy.get_action(state_tensor)
                    actions[agent_id] = action.item()
                elif algorithm == "IQL":
                    actions[agent_id] = agent.choose_action(state[agent_id])

            next_state, rewards, done, scores = env.step(actions)
            next_state = {agent_id: np.array(next_state[agent_id]).flatten() for agent_id in next_state}

            for agent_id, reward in rewards.items():
                if agent_id.startswith("team_1"):
                    episode_scores["team_1"] += reward
                elif agent_id.startswith("team_2"):
                    episode_scores["team_2"] += reward

            if done["__all__"]:
                if scores["team_1"] > scores["team_2"]:
                    results["team_1_wins"] += 1
                    win_status.append("Team_1")
                elif scores["team_2"] > scores["team_1"]:
                    results["team_2_wins"] += 1
                    win_status.append("Team_2")
                else:
                    results["draws"] += 1
                    win_status.append("Draw")
                break

            state = next_state

        scores_history.append(episode_scores)
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
    print(f"Draws: {results['draws']}")

    os.makedirs(log_dir, exist_ok=True)
    csv_file = os.path.join(log_dir, f"{algorithm}_scores.csv")
    with open(csv_file, "w") as f:
        f.write("Episode,Team_1_Score,Team_2_Score,Win_Status,Team_1_Total_Wins,Team_2_Total_Wins,Draws\n")
        cumulative_team_1_wins = 0
        cumulative_team_2_wins = 0
        cumulative_draws = 0
        for i, (s1, s2, win) in enumerate(zip(scores_team_1, scores_team_2, win_status)):
            if win == "Team_1":
                cumulative_team_1_wins += 1
            elif win == "Team_2":
                cumulative_team_2_wins += 1
            elif win == "Draw":
                cumulative_draws += 1
            f.write(f"{i + 1},{s1},{s2},{win},{cumulative_team_1_wins},{cumulative_team_2_wins},{cumulative_draws}\n")
    print(f"Test results saved in {csv_file}")

def visualize_movement(env, ax, step, episode, scores):
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
    env = CaptureTheFlagEnv(grid_size=grid_size, team_size=team_size)

    # Load MAPPO agents
    mappo_agents = {
        agent_id: MAPPOAgent(
            obs_shape=(env.grid_size[0], env.grid_size[1], env.num_agents),
            action_space=env.action_space.n
        ) for agent_id in env.positions.keys()
    }
    mappo_model_dir = "/home/elliot/Desktop/Shirish/PhD/Courses/MARL-ECS/Project/codes/logs/20241124-174906/models"
    load_mappo_models(mappo_agents, mappo_model_dir)
    test_agents(env, mappo_agents, algorithm="MAPPO", log_dir="test_logs_mappo_cahnge")

    # Q-learning agents
    q_table_file = "/home/elliot/Desktop/Shirish/PhD/Courses/MARL-ECS/Project/codes/q_tables_capture_flag.pkl"
    q_tables = load_q_tables(q_table_file)
    q_agents = {
        agent_id: QLearningAgent() for agent_id in env.positions.keys()
    }
    for agent_id in q_agents:
        q_agents[agent_id].load_q_table(q_tables[agent_id])
    test_agents(env, q_agents, algorithm="IQL", log_dir="test_logs_iql_change")
