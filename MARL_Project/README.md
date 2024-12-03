

---

# Decentralized Multi-Robot Navigation with Dynamic Obstacles

This repository implements the decentralized multi-robot navigation system for Autonomous Surface Vehicles (ASVs) using Distributional Reinforcement Learning (DRL), based on the paper *"Decentralized Multi-Robot Navigation for Autonomous Surface Vehicles with Distributional Reinforcement Learning"*.

## Objective

The primary objective of this research is to benchmark the original solution from the paper, focusing on static and dynamic obstacles in a marine environment. The system is first tested using the provided framework, then dynamic obstacles are introduced to assess the adaptability and performance. Based on these experiments, modifications are made, and the updated performance is reported, comparing the original and modified solutions.

## Requirements

Before running the code, ensure you have the following dependencies installed:

- Python 3.8+
- ubuntu 20.04
- gym 0.19.0
- TensorFlow / PyTorch (depending on the implementation)
- NumPy
- Matplotlib


You have to install the necessary dependencies in your env first to run this code 



## Folder Structure

The folder structure for this project is as follows:

```
/Multi_Robot_distributional_RL_Navigation
│
├── /config           # Contains the config files for training agents 
├── /marinenav_env     # Contains the environment code (modifications in `env/marinenav_env.py` for dynamic obstacles)
├── /marinenav_env/env/utils/robot.py   # Contains and handels the logic of Robots movements and there perception observation logic 
├── policy/agent.py          #Contains the combined code of handeling the reward and states logic for diffrent agents
├── policy/DQN.py            #Contains the code for the DQN algorithm 
├── policy/IQN.py            #Contains the code for the IQN algorithm
├── policy/reply_buffer.py            #Contains the code for the episodic steps reply states
├── /trained_data         # Stores training results, logs
├── scripts/plot_eval2.py            #Contains the code for the ploting the comparison plots btw DQN and IQN trained agents
├── scripts/visualise_eval_episode.py            #Contains the code for the visualizing the eval episodes of trained Models 
├── scripts/visualise_exp_episode.py    #Contains the code for the visualizing the exp episodes of evaluated Models from (experiment data) 
├── run_exp.py  #Contains the code for running the evalution experiment on trained agents with RVO and APF Policy
├── train_iqn_agent.py  #Contains the code for traning the agent 
└── README.md        # This file
```

## How to Run the Code

1. **Set up the environment:**

   Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. **Run the baseline experiment:**

   To run the baseline experiment (with static obstacles):

   ```bash
   python train_iqn_agent.py --config config/config_IQN.json 
   ```

   This will execute the original solution as described in the paper. The configuration will be loaded from `config_IQN.json`, and dynamic obstacles will be turned off if it is given false inside the config file.

3. **Run the experiment with dynamic obstacles:**

   After modifying the environment to include dynamic obstacles, you can run the experiment by setting the `dynamic_env` flag to `True` in json config file:

   ```bash
   python train_iqn_agent.py --config config/config_IQN.json 
   ```

   note : To train an DQN network you can use different config_DQN file inside the config folder or simply turn the use IQN flag inside json of IQN config file you can do both

4. **View Results:**

   Results will be saved in the `/training_data` directory, including logs, training statistics

---

## Changes Made for Dynamic Obstacles

### 1. **`agent.py` (Agent Class Modifications)**

- **Dynamic Obstacle Handling:**  
  The agent class has been updated to handle dynamic obstacles. A new function `update_dynamic_obstacles()` was added to check for moving obstacles and adjust the agent's actions accordingly.
  
  - Dynamic obstacles are now considered during the agent's decision-making process, where the agent adjusts its path based on the predicted movement of obstacles.
  - The reward function has been modified to provide additional rewards for successfully avoiding dynamic obstacles.

### 2. **`mrin_env.py` (Environment Modifications)**

- **Dynamic Environment Support:**  
  The environment has been modified to support dynamic obstacles, which move based on predefined patterns or random behaviors. Changes made include:
  
  - The environment class now initializes dynamic obstacles and updates their positions at each timestep.
  - A flag (`dynamic_env`) has been added to enable or disable dynamic obstacles in the environment.
  - The environment dynamically updates the positions of obstacles based on their velocity and direction.
  
- **Reward Structure:**  
  The reward structure has been updated to incorporate additional penalties for collisions with dynamic obstacles and bonuses for avoiding them. This allows the agent to learn optimal behavior when navigating around moving obstacles.

### 3. **`trainer.py` (Trainer Modifications)**

- **Training with Dynamic Obstacles:**  
  The trainer script was updated to handle experiments with dynamic environments. Key modifications include:
  
  - Addition of a flag (`dynamic_env`) to toggle between static and dynamic environments.
  - The reward function was updated to account for dynamic obstacles and their influence on the agent's actions during training.
  - The training process now allows for the evaluation of the agent's performance with dynamic obstacles and reports success rates accordingly.

---

## Experimentation and Benchmarking

### 1. **Baseline Experiment (Static Obstacles):**
   
   In the baseline experiment, the system is tested using static obstacles to evaluate the performance of the original decentralized navigation algorithm as described in the paper. The performance is benchmarked in terms of:
   - Collision avoidance
   - Time taken to reach the goal
   - Energy efficiency
   
### 2. **Experiment with Dynamic Obstacles:**

   Once the baseline has been established, dynamic obstacles are introduced into the environment. This is done by enabling the `dynamic_env` flag and observing how the agent performs when facing moving obstacles.
   
   Key modifications for dynamic obstacles:
   - Dynamic obstacles are introduced into the environment using pre-defined patterns or random motions.
   - The reward function has been updated to reflect the new goal of avoiding moving obstacles.

### 3. **Updated Reward Structure:**

   The reward function has been updated in the following ways:
   - **Positive reward** for successfully avoiding dynamic obstacles.
   - **Penalty** for collisions with dynamic obstacles.
   - **Bonus rewards** for efficient navigation and minimal time to goal achievement.
   
---

## Future Work

- Further improve the agent's adaptability to dynamic obstacles by introducing advanced reinforcement learning techniques like Proximal Policy Optimization (PPO) or Attention-based mechanisms.
- Test the system in larger and more complex environments with multiple dynamic obstacles.
- Explore the incorporation of temporal and spatial dynamics for better handling of unpredictable changes in the environment.

---