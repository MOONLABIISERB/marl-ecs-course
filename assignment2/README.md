# Grid-Based Box Pushing with RL

This repository contains two Reinforcement Learning (RL) implementations for solving a grid-based environment where an agent must push a box to a storage location while avoiding obstacles.

## Environment Description
- **Grid**: The environment is represented as a 7x6 grid.
  - `1`: Obstacle
  - `0`: Free space
  - `2`: Initial box position
  - `3`: Storage position (goal)
- **Agent**: The agent can move in four directions: UP, DOWN, LEFT, RIGHT.
- **Goal**: Push the box to the storage position.

## Approaches

### 1. Value Iteration (Dynamic Programming)
This approach uses **Value Iteration** to find the optimal policy for the agent.

#### Key Features:
- **Rewards**:
  - Step: `-1`
  - Goal: `+100` for moving the box to the storage position.
  - Invalid Move: `-100` if the agent pushes the box into an obstacle or invalid location.
- **Transition Function**: Defines how the environment changes in response to agent actions.
- **Optimal Policy**: Derived after calculating the optimal value function for all possible states using the Bellman equation.

### 1. Monte-Carlo (First Visit)
This approach uses **Monte Carlo first visit** to find the optimal policy for the agent.

#### Key Features
- **Epsilon-Greedy Policy**: Balances exploration and exploitation.
- **Loop Detection**: Detects repetitive behavior and breaks loops.
- **Monte Carlo Control**: Learns from episodes with first-visit method.
- **Limitations**: Suboptimal policy due to limited exploration, state-space size, and loop issues.
