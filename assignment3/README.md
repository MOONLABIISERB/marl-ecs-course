# Multi-Agent Pathfinding Using Q-Learning

This assignment demonstrates a solution to a **Multi-Agent Pathfinding Problem (MAPF)** using **Multi-Agent Rollout Q-Learning**. It includes simulations of agents navigating a maze to reach their respective destinations while avoiding obstacles and conflicts.

---

## Environment

### Question 1: Multi-Agent Maze Simulation
- **Maze**: Represented as a grid. Cells can be empty, walls, or destinations.
- **Agents**: Start from random valid positions.
- **Actions**:
  - Stay, Move Up, Down, Left, or Right.
- **Rewards**:
  - +10 for reaching the destination.
  - -1 penalty for each step taken.

### Question 2: Optimal Pathfinding with random initialization at each step
- Trained with same method and setup
---

## Key Features
1. **Q-Learning**:
   - Each agent learns to select actions that maximize its cumulative reward over time. Q-values are updated based on the agent’s experiences during the training episodes, considering the exploration-exploitation trade-off.
2. **Rollout Policy**:
   - The rollout process involves each agent following the optimal path determined by its Q-table. At each step, an agent either explores a random action or exploits the best-known action based on its learned Q-values.
3. **Multi-Agent Setup**:
   - Multiple agents are trained simultaneously with their own independent Q-tables, navigating through a shared environment. Agents update their Q-values based on their own actions and rewards, while also accounting for the presence of other agents.

---


### Results
-  Minimum time for agents to reach goals and visualization of the maze with agents' optimal paths are shown in the notebook.
