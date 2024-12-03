import sys
sys.path.insert(0, "/home/bhanu/Documents/Multi_Robot_Distributional_RL_Navigation")
# print(sys.path)
from env_visualizer import EnvVisualizer
import json
import numpy as np
from policy.agent import Agent
import matplotlib.pyplot as plt 



if __name__ == "__main__":
    filename = "experiment_data/exp_data_2024-11-27-17-20-21/exp_results.json"

    with open(filename,"r") as f:
        exp_data = json.load(f)

    
    # schedule_id = -1
    # agent_id = 0
    ep_id = 0

    schedule_id = 2  # or 2 (last schedule)

    for agent_id in range(len(exp_data['all_eval_configs_exp'][schedule_id])):
        # print(f"Length of episode list for agent {agent_id}: {len(exp_data['all_eval_configs_exp'][schedule_id][agent_id])}")
    
        colors = ["r","lime","cyan","orange","tab:olive","white","chocolate"]


        ev = EnvVisualizer()

        ev.env.reset_with_eval_config(exp_data["all_eval_configs_exp"][schedule_id][agent_id][ep_id])
        ev.init_visualize()

        ev.play_episode(trajectories=exp_data["all_trajectories_exp"][schedule_id][agent_id][ep_id],
                        colors=colors)
        plt.close()