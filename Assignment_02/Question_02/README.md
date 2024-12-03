## Overview
This project implements a Sokoban puzzle environment and solves it using two reinforcement learning approaches:

1. **Dynamic Programming (DP)** using Value Iteration.
2. **Monte Carlo (MC)** using an epsilon-greedy exploration strategy.

## Sokoban Puzzle Rules
- The agent moves horizontally or vertically and pushes boxes onto designated storage locations.
- The agent **cannot pull boxes**, nor can it push boxes into walls or other boxes.
- The puzzle is solved when all boxes are placed on their respective storage locations.

## Environment Details
- **Grid Size**: 6 x 7.
- **States**: Represented as a tuple of positions for the agent, box, and target, e.g., `(player_position, box_position, target_position)`.
- **Actions**:
  - `UP`: Move the agent one square upward.
  - `DOWN`: Move the agent one square downward.
  - `LEFT`: Move the agent one square left.
  - `RIGHT`: Move the agent one square right.
- **Rewards**:
  - `-1` for invalid actions or moves.
  - `10` for solving the puzzle by placing the box on the target.
- **Termination**:
  - When the box is placed on the target.
  - If the agent attempts invalid moves indefinitely.

## Implemented Methods

### Dynamic Programming (DP)
- **Algorithm**: Value Iteration.
- **Goal**: Compute the optimal policy by iteratively updating state values using the Bellman equation.
- **Details**:
  - Enumerates all possible states.
  - Updates the value function until convergence.
  - Derives the optimal policy from the computed values.

### Monte Carlo (MC)
- **Algorithm**: First-Visit Monte Carlo with Epsilon-Greedy Exploration.
- **Goal**: Learn an optimal policy by averaging the returns of visited state-action pairs.
- **Details**:
  - Generates episodes of gameplay to explore the state-action space.
  - Uses epsilon-greedy exploration to balance exploration and exploitation.
  - Decays epsilon over time for improved convergence.

## Results

| Metric               | Dynamic Programming Agent | Monte Carlo Agent |
|----------------------|---------------------------|-------------------|
| Average Reward       | -59.34                    | -96.17            |
| Average Steps        |  63.59                    |  96.59            |

### Observations
- **Dynamic Programming** is computationally intensive but provides a robust policy.
- **Monte Carlo** training converges faster due to its exploratory approach.
- Both methods perform well in solving the Sokoban puzzle but differ in training efficiency.


## How to Run

### Setup
1. Install Python 3.8+.
2. Install dependencies:  gym numpy tqdm

### Execution
1. Save the scripts (`main.py`, `sokoban_environment.py`, `dp_agent.py`, and `mc_agent.py`) into a directory.
2. Run the `main.py` script:

### Output
- Displays training progress for both Dynamic Programming and Monte Carlo agents.
- Prints evaluation results for both agents.

## Files

- **`main.py`**: Entry point for training and evaluating the agents.
- **`sokoban_environment.py`**: Sokoban puzzle environment implementation.
- **`dp_agent.py`**: Implementation of the Dynamic Programming agent.
- **`mc_agent.py`**: Implementation of the Monte Carlo agent.
