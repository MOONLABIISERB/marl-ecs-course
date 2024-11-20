# Multi-Agent Pathfinding (MAPF) Using Q-Learning and Rollouts

This project addresses the Multi-Agent Pathfinding (MAPF) challenge by employing **Q-Learning** and multi-agent rollouts. The objective is to minimize the **maximum completion time** for all agents to reach their respective destinations.

In an extended version of the problem, the maze environment introduces randomness for enhanced robustness, with agents starting from new positions in each episode while other parameters such as destinations and obstacles remain fixed.

## Key Features

### Multi-Agent Rollouts
- Agents collectively explore paths and improve their strategies over multiple episodes.
- Incorporates randomness in actions to promote exploration while learning from past decisions.
- Q-tables are updated iteratively to ensure convergence towards minimizing the maximum time taken by any agent.

### Agent Communication
- **Position Awareness:** Agents avoid collisions by considering the positions of others as part of their decision-making.
- **Shared Environment Constraints:** Agents navigate a common maze respecting static obstacles, walls, and cells occupied by other agents.
- **Collaborative Reward Structure:** Encourages minimizing the collective completion time, promoting implicit coordination between agents.

### Dynamic Environment (Bonus Feature)
- Randomized starting positions for agents in each episode to enhance adaptability and ensure robust policy learning.
- Obstacles and destinations remain constant to focus on path optimization strategies.

### Optimal Path Visualization
- Displays the learned optimal paths for each agent using arrows to represent movement direction.
- Highlights the efficient navigation strategies developed over successive training episodes.

## Reward Structure
- **Step Penalty:** Each step taken incurs a penalty of -1, discouraging inefficient routes.
- **Goal Reward:** An agent reaching its destination earns a reward of +10, incentivizing goal-oriented behavior.
- **Collision Avoidance:** Agents are prevented from moving into occupied cells, ensuring coordinated navigation and reducing delays.

## Significance of Multi-Agent Pathfinding
MAPF is a critical challenge in fields like robotics, warehouse automation, and gaming. Efficient solutions enable applications such as drone swarm coordination, autonomous vehicle routing, and managing large-scale agent systems.

---