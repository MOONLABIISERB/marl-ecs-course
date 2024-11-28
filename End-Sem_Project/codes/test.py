import pickle
import numpy as np
import matplotlib.pyplot as plt
from env2 import CaptureTheFlagEnv

class Agent:
    def __init__(self, num_actions=5):
        self.q_table = {}
        self.num_actions = num_actions

    def load_q_table(self, q_table):
        """Load the Q-table for the agent."""
        self.q_table = q_table

    def choose_action(self, state):
        """Choose the best action deterministically based on the Q-table."""
        state_key = tuple(state.flatten())
        if state_key in self.q_table:
            return np.argmax(self.q_table[state_key]) 
        else:
            return np.random.choice(range(self.num_actions))


def load_q_tables(filename):
    """Load Q-tables from the saved file."""
    with open(filename, "rb") as f:
        q_tables = pickle.load(f)
    return q_tables


def test_agents_with_visualization(env, agents, num_episodes=100, max_steps_per_episode=200):
    """Run testing episodes with real-time visualization."""
    results = {"team_1_wins": 0, "team_2_wins": 0, "draws": 0}
    rewards_log = []

    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].set_title("Agent Movements")
    ax[1].set_title("Episode Rewards")
    ax[1].set_xlabel("Episodes")
    ax[1].set_ylabel("Total Rewards")
    reward_history_team_1 = []
    reward_history_team_2 = []

    for episode in range(num_episodes):
        _, state = env.reset() 
        episode_rewards = {"team_1": 0, "team_2": 0}

        for step in range(max_steps_per_episode):
            actions = {
                agent_id: agent.choose_action(state[agent_id])
                for agent_id, agent in agents.items()
            }

            next_state, rewards, done, scores = env.step(actions)
            for agent_id, reward in rewards.items():
                if agent_id.startswith("team_1"):
                    episode_rewards["team_1"] += reward
                elif agent_id.startswith("team_2"):
                    episode_rewards["team_2"] += reward
            visualize_movement(env, ax[0], step, episode, scores)
            plt.pause(0.1)

            state = next_state

            if done["__all__"]:
                break
        if scores["team_1"] > scores["team_2"]:
            results["team_1_wins"] += 1
        elif scores["team_2"] > scores["team_1"]:
            results["team_2_wins"] += 1
        else:
            results["draws"] += 1

        rewards_log.append(episode_rewards)
        reward_history_team_1.append(episode_rewards["team_1"])
        reward_history_team_2.append(episode_rewards["team_2"])
        visualize_rewards(ax[1], reward_history_team_1, reward_history_team_2, episode + 1)

    print("\nTesting Results:")
    print(f"Team 1 Wins: {results['team_1_wins']}")
    print(f"Team 2 Wins: {results['team_2_wins']}")
    print(f"Draws: {results['draws']}")

    plt.ioff()
    plt.show()

    return rewards_log, results


def visualize_movement(env, ax, step, episode, scores):
    """Render the current state of the environment in real-time."""
    ax.clear()
    ax.set_xticks(np.arange(0, env.grid_size[1] + 1, 1))
    ax.set_yticks(np.arange(0, env.grid_size[0] + 1, 1))
    ax.grid(True)

    # Draw obstacles
    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle((obs[1], obs[0]), 1, 1, color="grey"))

    # Draw agents
    for agent, pos in env.positions.items():
        color = "cyan" if "team_1" in agent else "green"
        ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, color=color))
        ax.text(pos[1] + 0.5, pos[0] + 0.5, agent[-1], ha="center", va="center", color="black")

    # Draw flags
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
    # Load environment
    grid_size = (10, 10)
    team_size = 2
    env = CaptureTheFlagEnv(grid_size=grid_size, team_size=team_size)

    # Load trained Q-tables
    q_table_file = "/home/elliot/Desktop/Shirish/PhD/Courses/MARL-ECS/Project/codes/q_tables_capture_flag.pkl"  # Replace with your saved Q-table file
    q_tables = load_q_tables(q_table_file)

    # Initialize agents and load their Q-tables
    agents = {}
    for agent_id in env.positions.keys():
        agent = Agent()
        if agent_id in q_tables:
            agent.load_q_table(q_tables[agent_id])
        agents[agent_id] = agent

    # Test the agents with real-time visualization
    num_episodes = 100
    max_steps_per_episode = 150
    rewards_log, results = test_agents_with_visualization(env, agents, num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode)
