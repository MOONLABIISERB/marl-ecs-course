# Mid-Semester Exam: Travelling Salesman Problem using Q-Learning

This repository contains the implementation of a modified version of the Travelling Salesman Problem (TSP) solved using Q-Learning. The task involves finding the optimal path that maximizes the profit, which decays with the distance traveled.

---

## Table of Contents
1. [Problem Description](#problem-description)
2. [Environment Setup](#environment-setup)
3. [Q-Learning Solution](#q-learning-solution)
4. [Results](#results)
    - [Cumulative Rewards](#cumulative-rewards)
    - [Final Path Traversed](#final-path-traversed)
5. [Files in this Repository](#files-in-this-repository)
6. [How to Run](#how-to-run)
7. [Cumulative Reward Data](#cumulative-reward-data)

---

## Problem Description

In this project, we model the Travelling Salesman Problem (TSP) as a reinforcement learning task. The agent's goal is to visit all cities while maximizing the profit collected, which decays as the distance traveled increases. We solve this using Q-learning, where rewards are based on a linearly decaying profit system.

---

## Environment Setup

The TSP environment is defined in `tsp_env.py`, where:
- Cities are represented as nodes.
- The agent's objective is to traverse through all cities and return to the starting point while maximizing the total reward.

---

## Q-Learning Solution

The `q_learning_sol.py` implements the Q-learning algorithm to solve this problem. The Q-learning parameters, such as the learning rate, discount factor, and exploration strategy, are tuned to maximize the cumulative rewards over several episodes.

Key points of the solution:
- **State Representation**: Each state represents the current city and the set of unvisited cities.
- **Actions**: The agent can move to any unvisited city.
- **Rewards**: The reward is based on the remaining profit, which decays as the distance increases.
  
---

## Results

### Cumulative Rewards

The figure below shows the cumulative rewards obtained after each episode during training. As seen, the agent improves its policy over time but exhibits fluctuations due to the exploration-exploitation trade-off in Q-learning.

![Cumulative Rewards over Episodes](https://raw.githubusercontent.com/user/repo/main/episodic_cumulative_reward.png)

- The figure above illustrates the improvement in total rewards after each episode. 
- The agent learns the optimal strategy over time but explores suboptimal paths occasionally.

---

### Final Path Traversed

After training, the final path traversed by the agent is displayed below. The path is plotted based on the positions of cities, with red dots representing the cities and the blue line representing the path taken by the agent.

![Final Path Traversed](https://raw.githubusercontent.com/user/repo/main/path_traversal.png)

---

## Files in this Repository

- `tsp_env.py`: This file contains the implementation of the TSP environment, defining the cities, distances, and reward mechanism.
- `q_learning_sol.py`: This file implements the Q-learning algorithm for solving the TSP problem.
- `episodic_cumulative_reward.png`: This image shows the plot of cumulative rewards over episodes during training.
- `path_traversal.png`: This image represents the final path traversed by the agent after training.
- `README.md`: This file, providing an overview and instructions for running the project.

---

## How to Run

1. **Install Dependencies**: Ensure you have the following dependencies installed:
   ```bash
   pip install numpy matplotlib gymnasium
   ```

2. **Run the Q-learning solution**:
   ```bash
   python q_learning_sol.py
   ```

3. **Visualize Results**: The results will be saved as images (`episodic_cumulative_reward.png` and `path_traversal.png`), which can be viewed to assess the agent's performance.

---

## Cumulative Reward Data

Here is a breakdown of the cumulative rewards at specific episodes:

- Cumulative rewards after episode 0: **44.52**
- Cumulative rewards after episode 50: **160.66**
- Cumulative rewards after episode 100: **191.48**
- Cumulative rewards after episode 150: **145.21**
- Cumulative rewards after episode 200: **186.64**
- Cumulative rewards after episode 250: **-22.36**
- Cumulative rewards after episode 300: **158.65**
- Cumulative rewards after episode 350: **182.13**
- Cumulative rewards after episode 400: **75.89**
- Cumulative rewards after episode 450: **85.89**

---
### Analysis of Results

#### 1. **Cumulative Rewards Trend**
   - **Initial Episodes (0-50)**:
     - At the start of training, the cumulative reward is relatively low (44.52 after episode 0). The agent is still exploring and learning about the environment.
     - The rewards improve rapidly during the first 50 episodes, reaching **160.66**. This indicates that the agent is beginning to learn better policies through exploration and exploitation.
   
   - **Middle Episodes (100-300)**:
     - The cumulative reward peaks at **191.48** after 100 episodes, suggesting that the agent has found a near-optimal path that balances exploration and exploitation.
     - However, the rewards drop sharply in some episodes, such as **145.21** after episode 150 and even **-22.36** after episode 250. These fluctuations are common in Q-learning as the agent may take exploratory actions that lead to suboptimal paths.
     - By episode 300, the agent regains its performance with a reward of **158.65**. Despite fluctuations, the rewards consistently hover around a high value.

   - **Later Episodes (350-500)**:
     - The agent's performance stabilizes after 350 episodes, with rewards like **182.13** after episode 350. This indicates that the agent has learned a good policy but still occasionally explores less rewarding paths.
     - After episode 400, we see another drop to **75.89**, highlighting the dynamic nature of Q-learning as the agent continues to balance exploration and exploitation.
     - By episode 450, the cumulative reward increases again to **85.89**, but the fluctuations still suggest room for improvement.

   **Key Takeaways**:
   - The **peaks** in the reward trend indicate that the agent is learning a good policy for maximizing rewards in the TSP problem.
   - The **fluctuations** suggest that the exploration-exploitation trade-off is still affecting performance, causing the agent to take suboptimal paths occasionally.
   - The **negative reward** after episode 250 is particularly interesting. This could indicate that the agent explored a very suboptimal route, highlighting the randomness involved in the exploration process of Q-learning.

#### 2. **Final Path Traversal**

   - The path traversed by the agent shows a non-optimal sequence of moves, with no cities being revisited but long distances being traveled between cities.
   - The Q-learning algorithm, while effective, may struggle with global optimality in TSP-like problems, especially when rewards decay linearly with distance traveled.
   - In the current solution, the final path is not the most efficient route. The agent has learned to prioritize shorter paths, but it has not yet fully optimized its traversal strategy.
   - **Potential Improvements**:
     - Tuning the learning rate or exploration decay parameters could help the agent find a more optimal path.
     - Introducing **heuristics** or **domain-specific knowledge** into the reward function could guide the agent more effectively toward better routes.

#### 3. **Exploration vs. Exploitation**
   - The fluctuating rewards are a sign of a balance between **exploration** and **exploitation**. During the exploration phase, the agent sometimes chooses suboptimal actions to discover better future rewards.
   - The fact that cumulative rewards do not remain consistently high suggests that the current balance could be adjusted. The agent might benefit from a more aggressive exploitation strategy once it has gathered sufficient information about the environment.
   - **Epsilon Decay**: If the exploration rate (epsilon) decreases more slowly, the agent may take more time to explore potential paths and avoid early convergence on a suboptimal solution.
