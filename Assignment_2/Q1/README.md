# Sokoban Puzzle with Reinforcement Learning

## Overview

This project sets up a grid-world environment for the Sokoban puzzle, solved using reinforcement learning. In Sokoban, an agent (player) moves through a grid to push boxes into marked storage locations. The rules are:

- The agent can move up, down, left, or right, but it can't move through walls or boxes.
- The agent can only push boxes by walking into them and pushing them forward. It can't pull boxes.
- Boxes can't be pushed into walls or other boxes.
- The number of boxes matches the number of storage locations.
- The puzzle is solved when all the boxes are on their correct storage spots.

## Environment Details

**Grid Size**: The grid is 6 x 7.

**State**: Each position in the grid is a "state." For example, if the agent is at row 2, column 3, the state is represented as `(2, 3)`.

**Actions**: The agent can move in four directions: UP, DOWN, LEFT, or RIGHT. It can push boxes but not pull them. Some actions can get boxes stuck in places where they can't be moved anymore.

**Rewards**:
- The agent gets a penalty (-1) when a box is not in a storage spot.
- The agent gets no penalty (0) when a box is successfully placed on a storage spot.

**Game Over**:
- The game ends when all boxes are on storage spots.
- The game also ends if a box gets stuck in an unrecoverable spot (like a corner or against a wall).

## How We Solve the Puzzle

### 1. **Dynamic Programming (DP)**:
   Use either value iteration or policy iteration to solve the puzzle.

### 2. **Monte Carlo (MC)**:
   Use the Monte Carlo method with random starts. Compare two methods: First-Visit and Every-Visit Monte Carlo.

## Results

| Metric               | Dynamic Programming Agent | Monte Carlo Agent |
|----------------------|---------------------------|-------------------|
| **Training Time**     | 266.38 seconds           | 63.40 seconds      |
| **Average Reward**    | -97.81                   | -88.30            |
| **Average Steps**     | 98.03                    | 89.51             |

### Comparison

- **Training Time**: The Monte Carlo agent trains faster than the Dynamic Programming agent.
- **Average Reward**: Both agents got negative rewards, but the Monte Carlo agent performed better.
- **Average Steps**: The Monte Carlo agent used fewer steps to solve the puzzle compared to the Dynamic Programming agent.
