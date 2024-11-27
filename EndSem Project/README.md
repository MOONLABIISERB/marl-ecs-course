# Multi-Agent Grid Coverage Using Reinforcement Learning

## Project Overview
This project explores a **multi-agent reinforcement learning framework** to enable autonomous agents to collaboratively explore and completely cover a 10x10 grid. The agents operate in a dynamic environment with or without obstacles, aiming to visit every cell on the grid exactly once while optimizing their behavior using **Q-Learning**.

---

## Problem Statement
Design and implement a solution where multiple agents:
- Operate on a shared 10x10 grid environment.
- Navigate around randomly placed obstacles.
- Use reinforcement learning to collaboratively cover the grid.
- Ensure every non-obstacle cell is visited at least once while minimizing overlaps and collisions.

---

## Approach
The problem is formulated as a **Markov Decision Process (MDP)** and solved using a **Q-Learning algorithm**. 

### Key Components:
1. **Environment Setup**:
   - **Grid Size**: 10x10.
   - **Agents**: 3 agents with fixed starting positions: \((0,0), (9,0), (9,9)\).
   - **Obstacles**: 10 randomly placed static obstacles.
   - **Rewards**:
     - Positive for visiting new cells.
     - Negative for revisits or collisions.
     - Large bonus for complete grid coverage.

2. **Algorithm**:
   - **Q-Learning**:
     - Agents learn an optimal policy by iteratively updating Q-values based on state transitions and rewards.
   - **Epsilon-Greedy Exploration**:
     - Encourages a balance between exploration and exploitation during training.
   - **Independent Multi-Agent Learning**:
     - Each agent learns individually while interacting within the shared environment.

3. **Metrics**:
   - **Cumulative Rewards**: Tracks the total reward across episodes.
   - **Agent Coverage Map**: Visual representation of grid cells covered by each agent.
   - **Temporal Difference (TD) Errors**: Monitors learning progress for each agent.

---

## Experimentation
### Goals:
- Assess the performance of Q-Learning in achieving complete grid coverage.
- Evaluate the effect of obstacles on learning and agent coordination.
- Analyze reward patterns, convergence, and agent efficiency.

### Key Results:
1. **Grid Coverage**:
   - Agents successfully achieved partial grid coverage in most episodes.
   - Dynamic obstacle placement required agents to adapt their policies.

2. **Learning Trends**:
   - Cumulative rewards improved over episodes, indicating effective learning.
   - TD errors decreased as agents converged to optimal policies.

3. **Visualization**:
   - **Coverage Map**: Color-coded grid indicating which agent visited each cell.
   - **Reward Trends**: Line plot showing cumulative rewards across episodes and moving average of TD errors.
     

---

