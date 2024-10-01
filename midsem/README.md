# Modified TSP Environment with Q-Learning

This repository contains a custom RL environment simulating a variation of the **Traveling Salesman Problem (TSP)**, combined with a Q-learning agent. The agent's goal is to visit all targets and maximize profits, which decay over time. The environment and agent are designed to explore reinforcement learning in a complex decision-making scenario.

## Environment Overview

The environment simulates a TSP where:
- **Targets** are randomly generated within a defined area.
- **Profits** decrease as time passes, and visiting a target more than once incurs a heavy penalty.
- The agent must learn to navigate these targets.
  
## Q-Learning Agent

The agent uses Q-learning to optimize target selection based on cumulative rewards:
- **TD error **, which helps evaluate learning performance.
- A greedy action selection strategy using the Q-value table.


## Key Insights
- The Q-learning agent shows convergence after 4000-5000 episode for 10 targets, yielding reasonable performance.
- SARSA was initially tested but did not show satisfactory convergence behavior, hence I switched to a fully greedy Q-learning approach.
- Profits were shuffled periodically (at every 10th episode) to introduce dynamic complexity, which made learning more challenging 

## Plots

Two main metrics were plotted:
1. **Cumulative Reward vs Episode** – Tracks how well the agent performs over time.
2. **TD Error vs Episode** – Measures the temporal difference error over the course of training.
   
<img width="985" alt="Screenshot 2024-10-01 at 2 49 28 PM" src="https://github.com/user-attachments/assets/55d53fda-1446-42ca-ad10-b5212dfcce6d">

## Usage

To run the program:
1. Install dependencies: 
   ```bash
   pip install gymnasium numpy pandas matplotlib
