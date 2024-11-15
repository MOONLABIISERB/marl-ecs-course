# MARL Assignment-3

## Overview
This assignment focuses on training multi-agent systems using reinforcement learning to achieve efficient pathfinding in a grid world.

## Directory Structure
.
├── env2.py
├── env.py
├── evaluate.py
├── __init__.py
├── MARL - Assignment-3.pdf
├── __pycache__
│   ├── cac_network.cpython-312.pyc
│   ├── env2.cpython-312.pyc
│   ├── env.cpython-312.pyc
│   ├── gae.cpython-312.pyc
│   └── MAPPO.cpython-312.pyc
├── requirements.txt
├── rewards_and_steps.png
├── train.py
└── visualize.py

## Training Results
1. Rewards and Steps Over Episodes
The agents were trained over 500 episodes. The performance was measured in terms of cumulative rewards and the number of steps taken per episode.


Rewards: The rewards plot shows initial fluctuations as agents explore various actions. Over time, the total rewards stabilize, indicating agents' convergence to effective strategies.
Steps: Initially, agents take the maximum allowable steps as they explore. Over episodes, the steps decrease, reflecting more efficient paths as agents learn to optimize their movements.
2. Agent Performance and Completion Times
At the end of training, each agent was evaluated based on its time taken to reach the target and ranked accordingly. The evaluation results are as follows:

### **Final Rewards by Agent:**

agent_0: -2
agent_1: -2
agent_2: -2
agent_3: 31
Completion Times (in seconds):

agent_3: 5.11 seconds
agent_0: 6.69 seconds
agent_1: 6.69 seconds
agent_2: 6.69 seconds
Ranking of Agents by Time Taken
Rank 1: agent_3 with 5.11 seconds
Rank 2: agent_0 with 6.69 seconds
Rank 3: agent_1 with 6.69 seconds
Rank 4: agent_2 with 6.69 seconds
This ranking suggests that agent_3 consistently reached the target faster than the other agents.

The Q-tables (stored in q_tables.pkl) reflect the learned values for each state-action pair per agent. Analysis of these tables shows convergence in the policy, with agents preferring actions that minimize distance and avoid penalties.
## Usage of Q-Tables for Evaluation
To reuse the trained Q-tables for evaluation:

Load q_tables.pkl to initialize agents' policies.
Run evaluate_agents() in visualization.py to visualize agent movements based on learned Q-values.
This setup allows evaluation of the trained policies without retraining.

