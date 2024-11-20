import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
import random

class GridEnvironment:
    def __init__(self, agent_positions, goal_positions):
        self.agent_positions = agent_positions
        self.goal_positions = goal_positions
        self.num_agents = len(agent_positions)
        self.grid = np.zeros((10, 10))
        self.grid[:3, 5] = -1
        self.grid[2, 4] = -1
        self.grid[5, :3] = -1
        self.grid[4, 2] = -1
        self.grid[5, 7:] = -1
        self.grid[4, 7] = -1
        self.grid[7:, 4] = -1
        self.grid[7, 5] = -1

    def reset(self):
        """Resets the environment and returns initial state."""
        self.current_positions = self.agent_positions.copy()
        return self.current_positions

    def step(self, actions):
        """Executes actions for all agents and updates their positions."""
        proposed_positions = {}
        rewards = {}
        done = True

        for agent_id, action in actions.items():
            x, y = self.current_positions[agent_id]

            if (x, y) == self.goal_positions[agent_id]:
                proposed_positions[agent_id] = (x, y)
            else:
                if action == "UP":
                    nx, ny = x - 1, y
                elif action == "DOWN":
                    nx, ny = x + 1, y
                elif action == "LEFT":
                    nx, ny = x, y - 1
                elif action == "RIGHT":
                    nx, ny = x, y + 1
                else:  # "STAY"
                    nx, ny = x, y

                if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1] and self.grid[nx, ny] != -1:
                    proposed_positions[agent_id] = (nx, ny)
                else:
                    proposed_positions[agent_id] = (x, y)

        for agent_id, pos in proposed_positions.items():
            if list(proposed_positions.values()).count(pos) > 1:
                proposed_positions[agent_id] = self.current_positions[agent_id]

        for agent_id, (x, y) in self.current_positions.items():
            new_pos = proposed_positions[agent_id]
            self.current_positions[agent_id] = new_pos

            if new_pos == self.goal_positions[agent_id]:
                rewards[agent_id] = 10  # Goal reward
            else:
                gx, gy = self.goal_positions[agent_id]
                distance_before = abs(x - gx) + abs(y - gy)
                distance_after = abs(new_pos[0] - gx) + abs(new_pos[1] - gy)
                rewards[agent_id] = 1 if distance_after < distance_before else -1

                done = False

        return self.current_positions, rewards, done

    def render(self, step):
        """Visualizes the grid with agents and goals in the same Matplotlib window."""
        plt.clf()

        ax = plt.gca()

        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x, y] == -1:
                    ax.add_patch(Rectangle((y, self.grid.shape[0] - 1 - x), 1, 1, color="gray"))
                else:
                    ax.add_patch(Rectangle((y, self.grid.shape[0] - 1 - x), 1, 1, fill=False, edgecolor="black"))


        colors = ["blue", "yellow", "green", "purple"]
        for agent_id, (gx, gy) in self.goal_positions.items():
            plt.text(gy + 0.5, self.grid.shape[0] - 1 - gx + 0.5, "+", color=colors[agent_id % len(colors)],
                     ha="center", va="center", fontsize=14)

        for agent_id, (ax, ay) in self.current_positions.items():
            plt.text(ay + 0.5, self.grid.shape[0] - 1 - ax + 0.5, f"A{agent_id}", color=colors[agent_id % len(colors)],
                     ha="center", va="center", fontsize=12, bbox=dict(boxstyle="circle", facecolor="white"))

        plt.xlim(0, self.grid.shape[1])
        plt.ylim(0, self.grid.shape[0])
        plt.gca().invert_yaxis()
        plt.title(f"Step {step}")
        plt.axis("off")

        plt.pause(0.1)

class QLearningWithRolloutsPolicy:
    def __init__(self, goal_positions, actions=["UP", "DOWN", "LEFT", "RIGHT", "STAY"], alpha=0.5, gamma=0.9, epsilon=0.9, rollout_depth=10, rollout_count=11):
        self.goal_positions = goal_positions
        self.q_table = defaultdict(lambda: {action: 0 for action in actions})
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rollout_depth = rollout_depth 
        self.rollout_count = rollout_count 

    def state_to_key(self, state):
        return tuple(sorted(state.items()))

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.1):
        self.epsilon = max(self.epsilon * decay_rate, min_epsilon)

    def choose_action(self, state, agent_id, env):
        if random.random() < self.epsilon:
            return random.choice(["UP", "DOWN", "LEFT", "RIGHT", "STAY"])

        best_action = None
        best_value = float("-inf")

        for action in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]:
            total_reward = 0
            for _ in range(self.rollout_count):
                total_reward += self.perform_rollout(state, agent_id, action, env)

            avg_reward = total_reward / self.rollout_count
            if avg_reward > best_value:
                best_action = action
                best_value = avg_reward

        return best_action

    def perform_rollout(self, state, agent_id, action, env):
        simulated_env = GridEnvironment(env.agent_positions, env.goal_positions)
        simulated_env.current_positions = state.copy()

        total_reward = 0
        current_state = state.copy()
        for _ in range(self.rollout_depth):
            actions = {agent_id: action}
            for other_id in state:
                if other_id != agent_id:
                    actions[other_id] = random.choice(["UP", "DOWN", "LEFT", "RIGHT", "STAY"])

            next_state, rewards, _ = simulated_env.step(actions)
            total_reward += rewards[agent_id]
            current_state = next_state

            if current_state[agent_id] == self.goal_positions[agent_id]:
                break

        return total_reward


def main():
    agent_positions = {0: (1, 1), 1: (1, 8), 2: (8, 1), 3: (8, 8)}
    goal_positions = {0: (8, 5), 1: (4, 1), 2: (4, 8), 3: (1, 4)}

    env = GridEnvironment(agent_positions, goal_positions)
    policies = {agent_id: QLearningWithRolloutsPolicy(env.goal_positions) for agent_id in agent_positions}

    state = env.reset()
    env.render(step=0)
    done=False
    step_count = 0
    while done==False:
        actions = {agent_id: policies[agent_id].choose_action(state, agent_id, env) for agent_id in state}
        next_state, rewards, done = env.step(actions)

        state = next_state
        step_count += 1

        if step_count % 10 == 0:
            env.render(step=step_count)

        for agent_id in state:
            policies[agent_id].decay_epsilon()


    print(f"All agents reached their goals in {step_count} steps!")

if __name__ == "__main__":
    main()
