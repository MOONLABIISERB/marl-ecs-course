# Modified Travelling Salesman Problem with DQN

This repository contains an implementation of the Modified Travelling Salesman Problem (TSP) environment using a Deep Q-Network (DQN) reinforcement learning agent. The goal of the agent is to navigate through a set of targets to maximize profits, with the complexity introduced by profit decay over time.

## Overview

- **Environment**: The agent operates within a custom gymnasium environment designed for the Modified TSP.
- **Agent**: A DQN agent that learns to navigate the environment using experience replay and prioritized sampling.
- **Training**: The agent is trained over a specified number of episodes to improve its performance in visiting targets.

## Environment: `ModTSP`

The `ModTSP` class simulates a TSP scenario where:
- The agent must visit a defined number of targets within a maximum area.
- Profits associated with each target decay as time progresses.
- The environment reshuffles profits after a set number of episodes to maintain variability.

### Key Methods

- `__init__`: Initializes the environment, generating target locations and calculating distances between them.
- `reset`: Resets the environment to an initial state for a new episode.
- `step`: Executes the action taken by the agent, updating the state and calculating the associated reward.
- `_generate_points`: Generates random 2D coordinates for target locations.
- `_calculate_distances`: Computes the distance matrix between all target locations.
- `_get_rewards`: Calculates the reward based on the agent's actions and the targets visited.

## DQN Agent: `DQNAgent`

The `DQNAgent` class implements the DQN algorithm with the following components:
- **Neural Network**: A feedforward neural network that approximates the Q-values for each action.
- **Experience Replay Buffer**: Stores past experiences for training the agent, with prioritized sampling for more significant learning.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation by gradually decreasing the exploration rate over time.

### Key Methods

- `update_target_model`: Softly updates the target model weights for stable learning.
- `remember`: Stores experiences in the replay buffer.
- `act`: Chooses an action based on the current state, either randomly (exploration) or from the model (exploitation).
- `replay`: Samples experiences from the buffer to update the model based on calculated Q-values.

## Training the Agent

The training process is handled in the `train_dqn` function, which performs the following:
- Resets the environment for each episode.
- Collects rewards and updates the agent's memory.
- Calls the `replay` function to train the model after accumulating enough experiences.

### Running the Code

To execute the code, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
