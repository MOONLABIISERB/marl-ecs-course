# Multi-Agent Pathfinding Using Q-Learning

This project demonstrates a solution to a **Multi-Agent Pathfinding Problem (MAPF)** using **Q-Learning**, a reinforcement learning technique. It includes simulations of agents navigating a maze to reach their respective destinations while avoiding obstacles and conflicts.

---

## Problem Overview

### Objective
- Multiple agents start at randomized positions in a maze.
- Each agent must navigate to its specific destination.
- Agents should avoid colliding with walls, each other, or revisiting their positions.

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

### Question 2: Optimal Path Visualization
- **Visualization**: Dotted lines represent the optimal path for each agent, computed from trained Q-tables.
- **Output**: Shows agents' paths, destinations, and invalid moves (if any).

---

## Solution Technique: Q-Learning
- **Q-Learning**: A model-free RL algorithm used to train agents to find optimal paths.
  - **State**: Each agent's position in the maze.
  - **Action**: Chosen based on an epsilon-greedy policy.
  - **Reward**: Immediate feedback for each action (step penalty or goal reward).
  - **Q-Table**: Learned policy mapping states to actions.
- **Algorithm**:
  1. Randomly initialize Q-tables for each agent.
  2. Train agents for multiple episodes using trial-and-error.
  3. Update Q-values using the Bellman equation:
     \[
     Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_a Q(s', a) - Q(s, a)]
     \]
  4. Visualize optimal paths based on the trained Q-tables.

---

## Key Features
1. **Dynamic Maze Environment**:
   - Randomized agent starting positions.
   - Static wall configurations.
2. **Optimal Path Visualization**:
   - Graphical representation of agents' paths.
   - Dotted lines for directions, color-coded by agent.
3. **Efficient Training**:
   - Independent Q-tables for agents with conflict handling.

---

## How to Run
1. Define the maze dimensions, wall positions, and agent destinations.
2. Initialize the `MultiAgentMazeMAPF` environment.
3. Train agents using the `train_agents_mapf` function.
4. Visualize the optimal paths using the `visualize_optimal_paths_with_dotted_lines` function.

---

## Libraries Used
- **Python**: Numpy, Matplotlib, Collections

---

### Example Outputs
- Training results: Minimum time for agents to reach goals.
- Visualization: Graph of the maze with agents' optimal paths.
