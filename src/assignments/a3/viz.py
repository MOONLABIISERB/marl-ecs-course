import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import time
from assignments.a3.env import GridWorldEnv


def visualize_gridworld(env, max_steps=100, save_animation=False):
    """
    Create an animation of the gridworld environment using a random policy.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Colors for different agents
    agent_colors = ["blue", "green", "red", "purple", "orange"][: env.n_agents]

    # Run random policy and store all states
    obs, _ = env.reset()
    states = []
    states.append({"agent_positions": env.agent_positions.copy(), "step": 0})

    done = False
    step = 0
    total_reward = 0

    while not done and step < max_steps:
        # Random actions
        actions = [env.action_space.sample() for _ in range(env.n_agents)]
        obs, rewards, done, _, _ = env.step(actions)

        # Store state
        states.append({"agent_positions": env.agent_positions.copy(), "step": step + 1})

        total_reward += sum(rewards)
        step += 1

    def init():
        ax.clear()
        # Draw grid
        for i in range(env.grid_size + 1):
            ax.axhline(y=i, color="gray", linestyle="-", alpha=0.3)
            ax.axvline(x=i, color="gray", linestyle="-", alpha=0.3)
        return []

    def update(frame):
        ax.clear()

        # Draw grid
        for i in range(env.grid_size + 1):
            ax.axhline(y=i, color="gray", linestyle="-", alpha=0.3)
            ax.axvline(x=i, color="gray", linestyle="-", alpha=0.3)

        # Draw walls
        for wall in env.walls:
            wall_rect = patches.Rectangle(
                (wall[0] - 0.5, wall[1] - 0.5), 1, 1, facecolor="gray"
            )
            ax.add_patch(wall_rect)

        # Get current state
        current_state = states[frame]

        # Draw agents and their goals
        for i in range(env.n_agents):
            # Draw agent
            agent_pos = current_state["agent_positions"][i]
            agent_circle = plt.Circle(
                (agent_pos[0], agent_pos[1]),
                0.3,
                color=agent_colors[i],
                alpha=0.7,
                label=f"Agent {i+1}",
            )
            ax.add_patch(agent_circle)

            # Draw goal
            goal_pos = env.goal_positions[i]
            goal_square = patches.Rectangle(
                (goal_pos[0] - 0.4, goal_pos[1] - 0.4),
                0.8,
                0.8,
                facecolor="none",
                edgecolor=agent_colors[i],
                linestyle="--",
                linewidth=2,
            )
            ax.add_patch(goal_square)

        # Set plot limits and labels
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_aspect("equal")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(f'Step {current_state["step"]}')

        return []

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(states),
        init_func=init,
        blit=True,
        interval=500,  # 500ms between frames
    )

    if save_animation:
        anim.save("gridworld_animation.gif", writer="pillow")

    plt.show()

    return total_reward, step


def run_episode_with_rendering(env, max_steps=100):
    """
    Run a single episode with real-time rendering.
    """
    obs, _ = env.reset()
    total_reward = 0
    step = 0

    fig, ax = plt.subplots(figsize=(8, 8))
    agent_colors = ["blue", "green", "red", "purple", "orange"][: env.n_agents]

    done = False
    while not done and step < max_steps:
        # Clear and redraw
        ax.clear()

        # Draw grid
        for i in range(env.grid_size + 1):
            ax.axhline(y=i, color="gray", linestyle="-", alpha=0.3)
            ax.axvline(x=i, color="gray", linestyle="-", alpha=0.3)

        # Draw walls
        for wall in env.walls:
            wall_rect = patches.Rectangle(
                (wall[0] - 0.5, wall[1] - 0.5), 1, 1, facecolor="gray"
            )
            ax.add_patch(wall_rect)

        # Draw agents and goals
        for i in range(env.n_agents):
            # Draw agent
            agent_pos = env.agent_positions[i]
            agent_circle = plt.Circle(
                (agent_pos[0], agent_pos[1]),
                0.3,
                color=agent_colors[i],
                alpha=0.7,
                label=f"Agent {i+1}",
            )
            ax.add_patch(agent_circle)

            # Draw goal
            goal_pos = env.goal_positions[i]
            goal_square = patches.Rectangle(
                (goal_pos[0] - 0.4, goal_pos[1] - 0.4),
                0.8,
                0.8,
                facecolor="none",
                edgecolor=agent_colors[i],
                linestyle="--",
                linewidth=2,
            )
            ax.add_patch(goal_square)

        # Set plot properties
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_aspect("equal")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(f"Step {step}")

        plt.draw()
        plt.pause(0.5)  # 500ms pause between steps

        # Take random actions
        actions = [env.action_space.sample() for _ in range(env.n_agents)]
        obs, rewards, done, _, _ = env.step(actions)

        total_reward += sum(rewards)
        step += 1

    plt.close()
    return total_reward, step


# Example usage
if __name__ == "__main__":
    # Define walls
    walls = [
        [2, 2],
        [2, 3],
        [2, 4],  # Vertical wall
        [4, 4],
        [5, 4],
        [6, 4],  # Horizontal wall
    ]

    # Define goal positions
    goal_positions = [
        [7, 7],  # Goal for agent 1
        [7, 6],  # Goal for agent 2
        [6, 7],  # Goal for agent 3
    ]

    # Create environment
    env = GridWorldEnv(
        grid_size=8,
        n_agents=3,
        walls=walls,
        goal_positions=goal_positions,
        random_start=True,
    )

    # Choose visualization method:
    # 1. Animated visualization (saves as GIF)
    total_reward, steps = visualize_gridworld(env, max_steps=50, save_animation=True)
    print(f"Episode finished after {steps} steps with total reward: {total_reward:.2f}")

    # 2. Real-time rendering
    # total_reward, steps = run_episode_with_rendering(env, max_steps=50)
    # print(f"Episode finished after {steps} steps with total reward: {total_reward:.2f}")
