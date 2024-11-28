import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

class CaptureTheFlagEnv(gym.Env):
    """
    This is my capture the flag environment, here I have used the below logics
    - Teams will be able to defend the flag in their region for a distance specifed as depth, I used 3
    - Teams if defending captures the opponent then they are rewarded positively while another negatively, and the game is won by capturing team
    - If team reach opponents flag then also they are considered winner and the env resets
    """
    def __init__(self, grid_size=(10, 10), team_size=2):
        super(CaptureTheFlagEnv, self).__init__()
        self.grid_size = np.array(grid_size)
        self.team_size = team_size
        self.num_agents = team_size * 2  # Two teams
        self.action_space = spaces.Discrete(5)  # 5 types of movements, left, right, up, down, and stay
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size[0], grid_size[1], self.num_agents), dtype=np.float32
        )

        self.action_dict = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}  # movements elaborated
        self.obstacles = [(3, 4), (3, 5), (3, 6), (4, 4), (5, 6), (6, 4), (6, 5), (6, 6)]
        self.flags = {"team_1_flag": (0, 0), "team_2_flag": (9, 9)}
        self.flag_status = {"team_1_flag": None, "team_2_flag": None}  # Tracks which agent holds the flag
        self.scores = {"team_1": 0, "team_2": 0}
        self.positions = {}

        self.reset()

    def reset(self):
        """Reset positions of agents and flags."""
        self.positions = self._initialize_agent_positions()
        self.flag_status = {"team_1_flag": None, "team_2_flag": None}
        self.scores = {"team_1": 0, "team_2": 0}
        return self._get_observations()

    def _initialize_agent_positions(self):
        """Initialize agent positions within their respective halves."""
        occupied_positions = set(self.obstacles) | set(self.flags.values())
        positions = {}

        # Team 1 agents: Top-left half of the grid
        for i in range(self.team_size):
            while True:
                pos = (
                    np.random.randint(0, self.grid_size[0] // 2),  # Rows: Top half
                    np.random.randint(0, self.grid_size[1] // 2)   # Columns: Left half
                )
                if pos not in occupied_positions:  # agents are intialized in the positions that are either not in obstacles or flags or same of both agents of a team
                    positions[f"team_1_agent_{i}"] = pos
                    occupied_positions.add(pos)
                    break

        # Team 2 agents: Bottom-right half, same logic as team 1
        for i in range(self.team_size):
            while True:
                pos = (
                    np.random.randint(self.grid_size[0] // 2, self.grid_size[0]),  # Rows: Bottom half
                    np.random.randint(self.grid_size[1] // 2, self.grid_size[1])   # Columns: Right half
                )
                if pos not in occupied_positions:
                    positions[f"team_2_agent_{i}"] = pos
                    occupied_positions.add(pos)
                    break

        return positions


    def step(self, actions):
        """
        Execute a step based on agent actions.
        """
        rewards = {agent: 0 for agent in self.positions.keys()} 
        done = {"__all__": False}

        for agent, action in actions.items():
            action = self._validate_action(action, agent)
            current_pos = self.positions[agent]
            dx, dy = self.action_dict[action]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)

            if self._valid_position(next_pos):
                self.positions[agent] = next_pos
            if agent.startswith("team_1") and next_pos == self.flags["team_2_flag"]:
                self.flag_status["team_2_flag"] = agent
                rewards[agent] += 100
                self.scores["team_1"] += 100
                done["__all__"] = True
            elif agent.startswith("team_2") and next_pos == self.flags["team_1_flag"]:
                self.flag_status["team_1_flag"] = agent
                rewards[agent] += 100 
                self.scores["team_2"] += 100
                done["__all__"] = True
        if current_pos == next_pos:
            rewards[agent] -= 2
        # Defending logic
        rewards.update(self.defend(actions))
        _ ,next_state = self._get_observations()
        return next_state, rewards, done, self.scores

    # def defend(self, actions):
    #     """
    #     Implement defending logic:
    #     - Team 2 defends its flag when Team 1 enters its territory.
    #     - Collisions result in positive rewards for defenders and penalties for intruders.
    #     """
    #     rewards = {agent: 0 for agent in self.positions.keys()}
    #     center_row = self.grid_size[0] // 2

    #     for agent, action in actions.items():
    #         current_pos = self.positions[agent]
    #         dx, dy = self.action_dict[action]
    #         next_pos = (current_pos[0] + dx, current_pos[1] + dy)

    #         # Check if Team 1 enters Team 2's territory
    #         if agent.startswith("team_1") and next_pos[0] >= center_row:
    #             # Check for collision with Team 2 defenders
    #             for defender, defender_pos in self.positions.items():
    #                 if defender.startswith("team_2") and defender_pos == next_pos:
    #                     rewards[defender] += 5  # Reward for successful defense
    #                     rewards[agent] -= 5  # Penalty for getting caught
    #                     print(f"Collision! {agent} caught by {defender}")

    #     return rewards
    def defend(self, actions):
        """
        Implement defending logic for both teams:
        - Each team defends its flag when an opponent comes within a 3-block radius.
        - Collisions result in positive rewards for defenders and penalties for intruders.
        """
        rewards = {agent: 0 for agent in self.positions.keys()}
        defense_radius = 2  # Radius within which defenders become active

        # Define flag positions for each team
        flag_positions = {
            "team_1_flag": self.flags["team_1_flag"],
            "team_2_flag": self.flags["team_2_flag"]
        }

        for agent, action in actions.items():
            current_pos = self.positions[agent]
            dx, dy = self.action_dict[action]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)

            # Check if Team 1 agent is within defense radius of Team 2's flag
            if agent.startswith("team_1"):
                distance_to_flag = np.linalg.norm(np.array(next_pos) - np.array(flag_positions["team_2_flag"]))
                if distance_to_flag <= defense_radius:
                    for defender, defender_pos in self.positions.items():
                        if defender.startswith("team_2") and defender_pos == next_pos:
                            rewards[defender] += 25  # Reward for successful defense
                            rewards[agent] -= 25  # Penalty for being caught
                            print(f"Collision! {agent} caught by {defender}")

            # Check if Team 2 agent is within defense radius of Team 1's flag
            if agent.startswith("team_2"):
                distance_to_flag = np.linalg.norm(np.array(next_pos) - np.array(flag_positions["team_1_flag"]))
                if distance_to_flag <= defense_radius:
                    for defender, defender_pos in self.positions.items():
                        if defender.startswith("team_1") and defender_pos == next_pos:
                            rewards[defender] += 25  # Reward for successful defense
                            rewards[agent] -= 25  # Penalty for being caught
                            print(f"Collision! {agent} caught by {defender}")

        return rewards


    def _validate_action(self, action, agent):
        """Validate the given action."""
        if isinstance(action, tuple):
            action = action[0]  # Unpack tuple if action is nested
        try:
            action = int(action)
        except ValueError:
            action = 4  # Default to 'stay' action

        if action not in self.action_dict:
            action = 4  # Default to 'stay' action

        return action

    def _valid_position(self, pos):
        """Check if the position is valid (within bounds and not an obstacle)."""
        x, y = pos
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and pos not in self.obstacles

    def _get_observations(self):
        """Return the current state as observations for all agents."""
        grid = np.zeros(self.grid_size)
        observations = {}
        for agent,pos in self.positions.items():
            grid[pos[0], pos[1]] = 1
        
        # flag = self.flags["team_2_flag"] if agent.startswith("team_1") else self.flags["team_1_flag"]
        # observations[agent] = {
        #             "position": pos,
        #             "flag": flag,
        #             "obstacles": self.obstacles,
        #         }
            observations[agent] = grid
        return grid,observations

    def render(self, step_count):
        """Visualize the grid and agents accurately in the same figure."""
        plt.ion()  # Turn on interactive mode

        # Create figure and axis if not already created
        if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.set_xticks(np.arange(0, self.grid_size[1] + 1, 1))
            self.ax.set_yticks(np.arange(0, self.grid_size[0] + 1, 1))
            self.ax.grid(True)

        self.ax.clear()  # Clear the previous content in the figure

        # Draw obstacles
        for obs in self.obstacles:
            self.ax.add_patch(plt.Rectangle((obs[1], obs[0]), 1, 1, color="grey"))

        # Draw agents
        for agent, pos in self.positions.items():
            color = "cyan" if "team_1" in agent else "green"
            self.ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, color=color))
            self.ax.text(pos[1] + 0.5, pos[0] + 0.5, agent[-1], ha="center", va="center", color="black")

        # Draw flags
        for flag, pos in self.flags.items():
            color = "blue" if "team_1" in flag else "red"
            self.ax.add_patch(plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.4, color=color))

        # Set axis limits to match the grid size
        self.ax.set_xlim(0, self.grid_size[1])
        self.ax.set_ylim(0, self.grid_size[0])
        self.ax.set_aspect('equal')

        # Update the title with step count and scores
        self.ax.set_title(f"Step: {step_count} | Scores: {self.scores}")
        # plt.show()
        plt.pause(0.1)  # Pause for a brief moment to show the updated frame

    def close_render(self):
        plt.ioff()
        # plt.show()
        plt.close()


def greedy_policy(agent, positions, flags, grid_size):
    """Greedy policy for moving toward the target."""
    target = flags["team_2_flag"] if agent.startswith("team_1") else flags["team_1_flag"]
    current_pos = positions[agent]
    dx = target[0] - current_pos[0]
    dy = target[1] - current_pos[1]
    move = (np.sign(dx), 0) if abs(dx) > abs(dy) else (0, np.sign(dy))
    action_map = {v: k for k, v in {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),4:(0,0)}.items()}
    return action_map.get(move, 4)

def run_simulation(env, max_steps=50):
    """Run the simulation."""
    done = {"__all__": False}
    step_count = 0

    while not done["__all__"] :
            obs = env.reset()
            
            while step_count < max_steps:
                step_count += 1
                actions = {agent: greedy_policy(agent, env.positions, env.flags, env.grid_size) for agent in env.positions}
                obs, rewards, done, scores = env.step(actions)
                env.render(step_count)
            step_count = 0
            env.close_render()
    print("Simulation finished!")

# Run the simulation
env = CaptureTheFlagEnv(grid_size=(10, 10), team_size=2)
# 
run_simulation(env)
# 