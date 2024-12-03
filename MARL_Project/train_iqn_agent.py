



import os
import argparse
import itertools
from multiprocessing import Pool
import json
from datetime import datetime
import numpy as np
from marinenav_env.envs.marinenav_env import MarineNavEnv2
from policy.agent import Agent
from policy.trainer import Trainer
import torch

# Argument parser to accept command-line arguments
parser = argparse.ArgumentParser(description="Train IQN model")

parser.add_argument(
    "-C",
    "--config-file",
    dest="config_file",
    type=open,
    required=True,
    help="configuration file for training parameters",
)
parser.add_argument(
    "-P",
    "--num-procs",
    dest="num_procs",
    type=int,
    default=1,
    help="number of subprocess workers to use for trial parallelization",
)
parser.add_argument(
    "-D",
    "--device",
    dest="device",
    type=str,
    default="cpu",
    help="device to run all subprocesses, could only specify 1 device in each run"
)

# Function to generate all possible combinations of parameters
def product(*args, repeat=1):
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

# Function to format trial parameters
def trial_params(params):
    if isinstance(params,(str,int,float)):
        return [params]
    elif isinstance(params,list):
        return params
    elif isinstance(params, dict):
        keys, vals = zip(*params.items())
        mix_vals = []
        for val in vals:
            val = trial_params(val)
            mix_vals.append(val)
        return [dict(zip(keys, mix_val)) for mix_val in itertools.product(*mix_vals)]
    else:
        raise TypeError("Parameter type is incorrect.")

# Function to display the current training configuration
def params_dashboard(params):
    print("\n====== Training Setup ======\n")
    print("seed: ",params["seed"])
    print("total_timesteps: ",params["total_timesteps"])
    print("eval_freq: ",params["eval_freq"])
    print("use_iqn: ",params["use_iqn"])
    print("dynamic_env: ", params.get("dynamic_env", False))  # Add this to print dynamic-env status
    print("\n")

# Function to run a single trial
def run_trial(device,params):

    project_dir = "/home/bhanu/Documents/Multi_Robot_Distributional_RL_Navigation/training_data"
    exp_dir = os.path.join(project_dir,
                           "training_" + params["training_time"],
                           "seed_" + str(params["seed"]))
    
    # Create the necessary directories
    os.makedirs(exp_dir, exist_ok=True)  # Use exist_ok=True to avoid permission issues if the directory already exists

    param_file = os.path.join(exp_dir,"trial_config.json")
    with open(param_file, 'w+') as outfile:
        json.dump(params, outfile)

    # Check if the "dynamic-env" key exists and set the boolean accordingly
    dynamic_env = params.get("dynamic_env", False)

    print("Dynamic environment flag is set to:", dynamic_env)  # Added debugging log

    # Pass the dynamic_env boolean to the environment constructor
    # print("params[num_cooperativeparams]",param["training_schedule"]["num_cooperative"])

    train_env = MarineNavEnv2(seed=params["seed"],num_robots=params["training_schedule"]["num_cooperative"], schedule=params["training_schedule"], dynamic_env=dynamic_env)
    eval_env = MarineNavEnv2(seed=253, num_robots=params["training_schedule"]["num_cooperative"],dynamic_env=dynamic_env)  # Make sure eval_env also receives the dynamic_env parameter

    cooperative_agent = Agent(cooperative=True, device=device, use_iqn=params["use_iqn"], seed=params["seed"] + 100)

    if "load_model" in params:
        cooperative_agent.load_model(params["load_model"], "cooperative", device)

    trainer = Trainer(train_env=train_env,
                      eval_env=eval_env,
                      eval_schedule=params["eval_schedule"],
                      cooperative_agent=cooperative_agent,
                      non_cooperative_agent=None,
                     )

    # Save the evaluation configuration to the experiment directory
    trainer.save_eval_config(exp_dir)

    # Start learning process
    trainer.learn(total_timesteps=params["total_timesteps"],
                  eval_freq=params["eval_freq"],
                  eval_log_path=exp_dir)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.device == "gpu":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("CUDA not available, switching to CPU.")
            device = "cpu"
    else:
        device = args.device

    params = json.load(args.config_file)
    params_dashboard(params)
    training_schedule = params.pop("training_schedule")
    eval_schedule = params.pop("eval_schedule")

    trial_param_list = trial_params(params)

    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")

    # Run the trials either serially or in parallel
    if args.num_procs == 1:
        for param in trial_param_list:
            param["training_time"] = timestamp
            param["training_schedule"] = training_schedule
            param["eval_schedule"] = eval_schedule
            run_trial(device, param)
    else:
        with Pool(processes=args.num_procs) as pool:
            for param in trial_param_list:
                param["training_time"] = timestamp
                param["training_schedule"] = training_schedule
                param["eval_schedule"] = eval_schedule
                pool.apply_async(run_trial, (device, param))

            pool.close()
            pool.join()

    print("\n====== Training Complete ======\n")


### to train run the file : python train_rl_agents.py -C CONFIG_FILE [-P NUM_PROCS] [-D DEVICE]