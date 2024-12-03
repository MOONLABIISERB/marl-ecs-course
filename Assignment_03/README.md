# Multi-Agent Path Finding (MAPF) using Multi-Agent Reinforcement Learning

## Project Overview

This project implements a solution to the Multi-Agent Path Finding (MAPF) problem using multi-agent reinforcement learning (MARL). The task involves navigating multiple agents in a grid world where each agent must reach its goal while avoiding obstacles and other agents. The objective is to minimize the maximum time taken by any agent to reach its goal.

### Problem Description:
- **Multiple agents** are positioned on a grid with distinct start and goal positions.
- **Obstacles** are placed in the grid to block movement.
- Each agent can move **up, down, left, right, or stay in place** if blocked.
- A penalty of -1 is applied for each step taken by an agent until it reaches its goal.

### Assignment Goal:
- Minimize the **maximum time** taken by any agent to reach its goal by training agents using a multi-agent reinforcement learning algorithm.


## Installation

To run the code, you need to have the following dependencies installed:

- Python 3.7 or higher
- Required libraries: `numpy`, `matplotlib`, `pandas` (for visualization), and any reinforcement learning framework such as `gym` or custom implementation.

## Results

After running the code, the following results will be generated:

1.  Training completed! Minimum max time: 14


2. **Paths of Agents**:
   - The paths followed by each agent during the training phase will be printed to the console.
     ```
    Agent 1 path: [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (4, 5), (5, 5), (5, 6), (6, 6), (6, 7), (6, 8), (5, 8)]
    Agent 2 path: [(8, 1), (8, 2), (7, 2), (7, 3), (6, 3), (5, 3), (5, 4), (5, 5), (5, 6), (4, 6), (3, 6), (2, 6), (1, 6), (1, 5)]
    Agent 3 path: [(8, 8), (7, 8), (7, 7), (6, 7), (6, 6), (6, 5), (6, 4), (6, 3), (6, 2), (6, 1), (5, 1)]
    Agent 4 path: [(1, 8), (2, 8), (3, 8), (3, 7), (3, 6), (3, 5), (3, 4), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (8, 4)]
     ```

3. **Training Progress Plot**:
![Graph plot](image.png)

4. **Grid with Agent Paths**:
![MAP](https://github.com/MOONLABIISERB/marl-ecs-course/blob/abhinav_20008/Assignment_03/MAP_plot.png)
