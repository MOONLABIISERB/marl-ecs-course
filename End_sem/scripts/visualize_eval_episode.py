import sys
sys.path.insert(0, r"C:\Users\Pavan\Desktop\Multi_Robot_Distributional_RL_Navigation-main")
import env_visualizer
import os

if __name__ == "__main__":
    dir = r"C:\Users\Pavan\Desktop\Multi_Robot_Distributional_RL_Navigation-main\Results3\DQN\training_2024-11-26-16-16-41\seed_9"

    eval_configs = os.path.join(dir, "eval_configs.json")
    evaluations = os.path.join(dir, "evaluations.npz")

    ev = env_visualizer.EnvVisualizer(seed=231)

    # colors = ["r", "lime", "cyan", "orange", "tab:olive", "white", "chocolate"]
    colors = [
    "r", "lime", "cyan", "orange", "tab:olive", "white", "chocolate",
    "blue", "purple", "yellow", "pink", "gold", "teal"
]

    eval_id = 29
    eval_episode = 40

    ev.load_eval_config_and_episode(eval_configs, evaluations)
    ev.play_eval_episode(eval_id, eval_episode, colors)
# import sys
# sys.path.insert(0, r"C:\\Users\\Pavan\Desktop\\Multi_Robot_Distributional_RL_Navigation-main")
# import env_visualizer
# import os

# if __name__ == "__main__":
#     dir = r"C:\Users\Pavan\Desktop\Multi_Robot_Distributional_RL_Navigation-main\Results3\DQN\training_2024-11-26-16-16-41\seed_9"

#     eval_configs = os.path.join(dir, "eval_configs.json")
#     evaluations = os.path.join(dir, "evaluations.npz")

#     ev = env_visualizer.EnvVisualizer(seed=231)

#     colors = ["r", "lime", "cyan", "orange", "tab:olive", "white", "chocolate"]

#     eval_id = 10
#     eval_episode = 40

#     ev.load_eval_config_and_episode(eval_configs, evaluations)
#     ev.play_eval_episode(eval_id, eval_episode, colors)

