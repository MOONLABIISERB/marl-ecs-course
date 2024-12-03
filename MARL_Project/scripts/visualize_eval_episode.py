

import numpy as np
import os
import matplotlib.pyplot as plt  # Assuming matplotlib is used for plotting

sys.path.insert(0, "/home/bhanu/Documents/Multi_Robot_Distributional_RL_Navigation")
import env_visualizer

if __name__ == "__main__":
    dir = "training_data/training_2024-11-26-17-43-29/seed_9"

    eval_configs = os.path.join(dir, "eval_configs.json")
    evaluations = os.path.join(dir, "evaluations.npz")
    print(evaluations)
    ev = env_visualizer.EnvVisualizer(seed=253)

    colors = ["r", "lime", "cyan", "orange", "tab:olive", "white", "chocolate"]

    # Load eval configuration and evaluation data
    ev.load_eval_config_and_episode(eval_configs, evaluations)
    eval_data = np.load(evaluations, allow_pickle=True)
    print(eval_data)
    # Assuming one of the arrays in evaluations.npz holds the number of episodes
    total_episodes = len(eval_data[eval_data.files[0]])  # Access any array, assuming they all share the same length

    print(f"Total episodes available: {total_episodes}")
    delay_time = .2
    # Loop through all episodes one by one
    eval_id = -1  # Use -1 to select the most recent evaluation
    for eval_episode in range(total_episodes):
        print(f"Visualizing episode {eval_episode+1}/{total_episodes}")
        ev.play_eval_episode(eval_id, eval_episode, colors,delay=delay_time)

        # Close the previous plot to avoid memory issues
        plt.close()  # This will close the previous plot and free up resources
