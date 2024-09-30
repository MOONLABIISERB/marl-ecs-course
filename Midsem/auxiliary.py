import matplotlib.pyplot as plt
import numpy as np

import torch
from modTSP import ModTSP
from agent import Agent, DQN

import os


def plot_tsp_path(locations, path, actions, folder="path"):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    locations = np.array(locations)
    path = np.array(path)

    for step in range(len(path)):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot all locations
        ax.scatter(locations[:, 0], locations[:, 1], c="blue", s=50)

        # Add location labels
        for i, loc in enumerate(locations):
            ax.annotate(f"{i}", (loc[0], loc[1]), xytext=(5, 5), textcoords="offset points")

        # Plot the path up to the current step
        for i in range(1, step + 1):
            ax.plot(path[i - 1 : i + 1, 0], path[i - 1 : i + 1, 1], c="red", linewidth=2, zorder=1)
            # Add step number and action label
            midpoint = (path[i - 1] + path[i]) / 2
            ax.text(midpoint[0], midpoint[1], f"{i}:{actions[i-1]}", color="darkred", fontweight="bold", ha="center", va="bottom")

        # Highlight the current position
        ax.plot(path[step, 0], path[step, 1], "go", markersize=10, zorder=2)

        # Highlight start and end
        ax.plot(path[0, 0], path[0, 1], "go", markersize=10, zorder=2, label="Start")
        if step == len(path) - 1:
            ax.plot(path[-1, 0], path[-1, 1], "ro", markersize=10, zorder=2, label="End")

        ax.set_xlim(locations[:, 0].min() - 1, locations[:, 0].max() + 1)
        ax.set_ylim(locations[:, 1].min() - 1, locations[:, 1].max() + 1)
        ax.set_title(f"TSP Path - Step {step}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.grid(True)
        ax.legend()

        # Save the figure
        plt.savefig(os.path.join(folder, f"step_{step:02d}.png"))
        plt.close(fig)


def run_inference(model_path: str, num_targets: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ModTSP(num_targets)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Load the saved model
    model = DQN(state_dim, n_actions).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    path = [env.locations[0]]
    actions = []

    print("initial profits", env.initial_profits)

    while not done and steps < num_targets - 1:
        state = obs
        visited_states = state[-10:]
        visited_states[0] = 1
        print(visited_states)
        unvisited_states = [i for i, s in enumerate(visited_states) if s == 0]

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = model.forward(state_tensor)

        # Mask Q-values of visited states to a very low value
        q_values_masked = q_values.clone()
        for idx, visited in enumerate(visited_states):
            if visited == 1:
                q_values_masked[0, idx] = float("-inf")

        action = q_values_masked.argmax().item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        path.append(env.locations[action])
        actions.append(action)

        print(f"Step {steps}: Action {action}, Reward {reward:.2f}, Total Reward {total_reward:.2f}")
        print(f"Distance travelled: {info['distance_travelled']:.2f}, Total distance: {info['total_distance']:.2f}")
        print(f"Current profits: {info['current_profits']}")
        print("---")

    print(f"Inference completed. Total reward: {total_reward:.2f}, Total steps: {steps}")

    plot_tsp_path(env.locations, path, actions)
