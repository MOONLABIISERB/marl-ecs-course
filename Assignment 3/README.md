# 🛤️ Multi-Agent Pathfinding (MAPF) Using Q-Learning  

This project addresses the **Multi-Agent Pathfinding (MAPF)** problem using **Q-Learning** and multi-agent rollouts. The goal is to minimize the maximum time any agent takes to reach its destination. A bonus feature adds dynamic environments by randomizing agents' starting positions in each episode.  

---

## ✨ Features  

- 🤖 **Multi-Agent Rollouts**: Simulations where agents collectively explore and refine strategies over multiple episodes.  
- 💬 **Implicit Communication**: Agents avoid collisions by incorporating shared environment constraints into decision-making.  
- 🌍 **Dynamic Environments**: Randomized starting positions ensure robust learning (bonus feature).  
- 📈 **Visualization**: Optimal paths for each agent are visualized with directional arrows.  

---

## 📋 Problem Overview  

### 🚶 Multi-Agent Rollouts  
Agents learn to navigate the maze by:  
1. 🔍 **Exploring**: Taking random actions to discover the environment.  
2. 📚 **Learning**: Updating Q-tables based on rewards and penalties.  
3. 🤝 **Collaborating**: Coordinating implicitly to minimize the maximum time across all agents.  

### 💡 Implicit Communication  
Agents communicate indirectly through:  
- 👀 **Position Awareness**: Avoiding collisions by observing the shared environment.  
- 🚧 **Environmental Constraints**: Respecting walls and occupied cells.  
- 🎯 **Shared Rewards**: Encouraging teamwork by using a common reward structure.  

---

## 🎯 Reward Structure  

| **Event**              | **Reward/Penalty** |  
|-------------------------|--------------------|  
| ➡️ Step Taken           | -1                |  
| 🏁 Destination Reached  | +10               |  
| 🚫 Collision Avoidance  | Implicit (blocked move) |  

The reward structure drives agents to:  
- Reduce unnecessary steps.  
- Efficiently reach their destinations.  
- Avoid collisions and navigate collaboratively.  

---
## Code

The Code is contained in the marl_assignment_three.ipynb

---

## Results

### Maze
Below is a maze generated during training:

![Screenshot 2024-11-20 223437](https://github.com/user-attachments/assets/8057f15f-160c-43cf-8949-e0eb74ab33a4)


### Optimal Path Visualization
After training, the optimal paths for each agent are visualized below. Each path is represented by arrows indicating the direction of movement, with destinations marked as colored crosses:

![Screenshot 2024-11-20 223625](https://github.com/user-attachments/assets/db97944c-b219-4860-b444-24bf20110174)


