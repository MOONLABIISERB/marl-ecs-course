import sys
sys.path.insert(0,"../")
import env_visualizer
import json
import numpy as np
from policy.agent import Agent

if __name__ == "__main__":
    filename = r"C:\Users\Pavan\Desktop\Multi_Robot_Distributional_RL_Navigation-main\Results\training_2024-10-24-10-52-06\seed_9\dqn_network_params.pth"

    with open(filename,"r") as f:
        exp_data = json.load(f)


    schedule_id = -1
    agent_id = 0
    ep_id = 0
    
    colors = ["r","lime","cyan","orange","tab:olive","white","chocolate"]


    ev = env_visualizer.EnvVisualizer()

    ev.env.reset_with_eval_config(exp_data["all_eval_configs_exp"][schedule_id][agent_id][ep_id])
    ev.init_visualize()

    ev.play_episode(trajectories=exp_data["all_trajectories_exp"][schedule_id][agent_id][ep_id],
                    colors=colors)
    