import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from env2 import CaptureTheFlagEnv  # Import your environment

class Agent:
    def __init__(self, num_actions=5, alpha=0.1, gamma=0.995, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.same_position_count = 0  # Track how long the agent stays in the same position

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.num_actions))
        return np.argmax(self.q_table.get(state, np.zeros(self.num_actions)))

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)
        
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

class CaptureTheFlagIQLTrainer:
    def __init__(self, env, num_episodes=500, max_steps_per_episode=200):
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.agents = {agent_id: Agent() for agent_id in env.positions.keys()}
        self.logs = {"episode_rewards": [], "max_times": []}

        # Visualization setup
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 1, figsize=(6, 10))
        self.ax[0].set_title("Agent Movements")
        self.ax[1].set_title("Episode Rewards")
        self.ax[1].set_xlabel("Episodes")
        self.ax[1].set_ylabel("Total Rewards")
        self.reward_history = []

    # def train(self):
    #     for episode in range(self.num_episodes):
    #         _,state = self.env.reset()
    #         total_rewards = {agent_id: 0 for agent_id in self.env.positions.keys()}
    #         max_time = 0

    #         # Initialize position tracker with hashable state representation
    #         position_tracker = {agent_id: self.get_state_repr(state[agent_id]) for agent_id in state}

    #         print(f"Starting Episode {episode + 1}/{self.num_episodes}")

    #         for step in range(self.max_steps_per_episode):
    #             actions = {}
    #             for agent_id, agent in self.agents.items():
    #                 agent_state = self.get_state_repr(state[agent_id])
    #                 actions[agent_id] = agent.choose_action(agent_state)

    #             next_state, rewards, done, _ = self.env.step(actions)

    #             for agent_id, agent in self.agents.items():
    #                 agent_state = self.get_state_repr(state[agent_id])
    #                 next_agent_state = self.get_state_repr(next_state[agent_id])

    #                 # Penalty for staying in the same position
    #                 if position_tracker[agent_id] == next_agent_state:
    #                     rewards[agent_id] -= 2  # Staying penalty
    #                     agent.same_position_count += 1
    #                 else:
    #                     agent.same_position_count = 0

    #                 position_tracker[agent_id] = next_agent_state
    #                 total_rewards[agent_id] += rewards[agent_id]

    #                 # Update Q-values
    #                 agent.update_q_value(agent_state, actions[agent_id], rewards[agent_id], next_agent_state)

    #             # Visualize agent movement
    #             self.visualize_movement(step)

    #             state = next_state
    #             max_time = step + 1

    #             if done["__all__"]:
    #                 print("All agents reached their goals.")
    #                 break

    #         self.logs["episode_rewards"].append(sum(total_rewards.values()))
    #         self.logs["max_times"].append(max_time)

    #         # Update reward visualization
    #         self.visualize_rewards(episode + 1)

    #     self.save_q_tables()
    #     self.save_logs()
    #     plt.ioff()
    #     plt.show()
    def train(self):
        """Train agents using Independent Q-Learning."""
        for episode in range(self.num_episodes):
            _, state = self.env.reset()
            total_rewards = {"team_1": 0, "team_2": 0}  # Separate rewards for teams
            max_time = 0

            # Initialize position tracker with hashable state representation
            position_tracker = {agent_id: self.get_state_repr(state[agent_id]) for agent_id in state}

            print(f"Starting Episode {episode + 1}/{self.num_episodes}")

            for step in range(self.max_steps_per_episode):
                actions = {}
                for agent_id, agent in self.agents.items():
                    agent_state = self.get_state_repr(state[agent_id])
                    actions[agent_id] = agent.choose_action(agent_state)

                next_state, rewards, done, _ = self.env.step(actions)

                for agent_id, agent in self.agents.items():
                    agent_state = self.get_state_repr(state[agent_id])
                    next_agent_state = self.get_state_repr(next_state[agent_id])

                    # Penalty for staying in the same position
                    if position_tracker[agent_id] == next_agent_state:
                        rewards[agent_id] -= 2  # Staying penalty
                        agent.same_position_count += 1
                    else:
                        agent.same_position_count = 0

                    position_tracker[agent_id] = next_agent_state

                    # Update team-specific rewards
                    if agent_id.startswith("team_1"):
                        total_rewards["team_1"] += rewards[agent_id]
                    elif agent_id.startswith("team_2"):
                        total_rewards["team_2"] += rewards[agent_id]

                    # Update Q-values
                    agent.update_q_value(agent_state, actions[agent_id], rewards[agent_id], next_agent_state)

                # Visualize agent movement
                self.visualize_movement(step)

                state = next_state
                max_time = step + 1

                if done["__all__"]:
                    print("All agents reached their goals.")
                    break

            self.logs["episode_rewards"].append(sum(total_rewards.values()))
            self.logs.setdefault("team_1_rewards", []).append(total_rewards["team_1"])
            self.logs.setdefault("team_2_rewards", []).append(total_rewards["team_2"])
            # print("Team 1 reward : " ,self.logs['team_1_rewards'])
            # print("Team 2 reward : " ,self.logs['team_1_rewards'])
            self.logs["max_times"].append(max_time)

            # Update reward visualization
            self.visualize_rewards(episode + 1)

        self.save_q_tables()
        self.save_logs()
        plt.ioff()
        plt.show()


    def get_state_repr(self, observation):
        """Convert observation into a hashable state representation."""
        return tuple(observation.flatten())  # Flatten into a tuple for Q-table indexing

    def save_q_tables(self, filename="q_tables_capture_flag.pkl"):
        q_tables = {agent_id: agent.q_table for agent_id, agent in self.agents.items()}
        with open(filename, "wb") as f:
            pickle.dump(q_tables, f)
        print("Q-tables saved to", filename)

    def save_logs(self, filename="training_logs_capture_flag.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.logs, f)
        print("Training logs saved to", filename)

    def visualize_movement(self, step):
        self.ax[0].clear()
        self.ax[0].set_xticks(np.arange(0, self.env.grid_size[1] + 1, 1))
        self.ax[0].set_yticks(np.arange(0, self.env.grid_size[0] + 1, 1))
        self.ax[0].grid(True)

        # Draw obstacles
        for obs in self.env.obstacles:
            self.ax[0].add_patch(plt.Rectangle((obs[1], obs[0]), 1, 1, color="grey"))

        # Draw agents
        for agent, pos in self.env.positions.items():
            color = "cyan" if "team_1" in agent else "green"
            self.ax[0].add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, color=color))
            self.ax[0].text(pos[1] + 0.5, pos[0] + 0.5, agent[-1], ha="center", va="center", color="black")

        # Draw flags
        for flag, pos in self.env.flags.items():
            color = "blue" if "team_1" in flag else "red"
            self.ax[0].add_patch(plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.4, color=color))

        self.ax[0].set_xlim(0, self.env.grid_size[1])
        self.ax[0].set_ylim(0, self.env.grid_size[0])
        self.ax[0].set_aspect("equal")
        self.ax[0].set_title(f"Step {step + 1}")
        plt.pause(0.01)
    # def visualize_rewards(self, episode):
    #     """Visualize rewards in real-time for both teams with dynamic scaling."""
    #     self.ax[1].clear()  # Clear previous plot to prevent overlaps
        
    #     # Plot team rewards
    #     if "team_1_rewards" in self.logs and len(self.logs["team_1_rewards"]) > 0:
    #         self.ax[1].plot(
    #             range(1, episode + 1), self.logs["team_1_rewards"], label="Team 1 Rewards", color="cyan"
    #         )
    #     if "team_2_rewards" in self.logs and len(self.logs["team_2_rewards"]) > 0:
    #         self.ax[1].plot(
    #             range(1, episode + 1), self.logs["team_2_rewards"], label="Team 2 Rewards", color="green"
    #         )

    #     # Dynamic y-axis scaling
    #     all_rewards = self.logs["team_1_rewards"] + self.logs["team_2_rewards"]

    #     if all_rewards:
    #         self.ax[1].set_ylim(0, max(all_rewards) * 1.1)  # Scale y-axis to fit rewards

    #     self.ax[1].set_xlabel("Episodes")
    #     self.ax[1].set_ylabel("Rewards")
    #     self.ax[1].set_title("Episode Rewards")
    #     self.ax[1].legend()  # Add legend once
    #     plt.pause(0.01)


    # def visualize_rewards(self, episode):
    #     self.ax[1].plot(range(1, episode + 1), self.logs["episode_rewards"], label="Total Rewards")
    #     self.ax[1].legend()
    #     plt.pause(0.01)
    def visualize_rewards(self, episode):
        """Visualize rewards in real-time for both teams."""
        self.ax[1].clear()  # Clear previous plot to prevent overlaps
        self.ax[1].plot(
            range(1, episode + 1), self.logs["team_1_rewards"], label="Team 1 Rewards", color="cyan"
        )
        self.ax[1].plot(
            range(1, episode + 1), self.logs["team_2_rewards"], label="Team 2 Rewards", color="green"
        )
        self.ax[1].set_xlabel("Episodes")
        self.ax[1].set_ylabel("Rewards")
        self.ax[1].set_title("Episode Rewards")
        self.ax[1].legend()  # Add legend once
        plt.pause(0.01)


if __name__ == "__main__":
    grid_size = (10, 10)
    team_size = 2
    num_episodes = 10000

    env = CaptureTheFlagEnv(grid_size=grid_size, team_size=team_size)
    trainer = CaptureTheFlagIQLTrainer(env, num_episodes=num_episodes, max_steps_per_episode=150)
    trainer.train()
