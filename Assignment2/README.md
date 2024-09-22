# QUESTION 1

In the first question MDP has been formulated as follows :-

- 1. The states consist of a 2 tuple, one is the current node and other is a list containing all visited nodes.
- 2. Actions are all possible targets i.e. (0,1,2,3,4,5) for the case where we have 6 targets.
- 3. 0 is the start node i.e. (0, ()) is the start state.
- 4. The transition probabilty is always 1 for a given state and action as the environment dynamics are deterministic- the agent always goes to a fixed state given any state (current and visited states pair) and action.
- 5. If the agent goes from state i to state j it gets a reward equal to negative of the distance between them, and it gets a large negative reward to return to already visited node.

# Dynamic Programming vs. Monte Carlo in Reinforcement Learning

This outlines the key differences between **Dynamic Programming (DP)** and **Monte Carlo (MC)** methods in reinforcement learning :

## 1. Requirements

- **DP**: Requires a complete model of the environment, including transition probabilities `P(s'|s, a)` and reward function `R(s, a, s')`.
- **MC**: Does not require explicit knowledge of the environment model. It learns directly from sampled episodes (state-action-reward sequences).

## 2. State-Action Value Updates

### DP (e.g., Value Iteration)
- Updates state values using the Bellman equation.
- Iterates over all states and actions, updating values based on expected rewards and the values of successor states.

# QUESTION 2

This repository showcases two different Reinforcement Learning (RL) approaches to solve a grid-world task where an agent needs to push a box to a designated goal position while avoiding obstacles.

## Environment Overview
- **Grid Layout**: The environment is modeled as a 7x6 grid.
  - `1`: Represents obstacles the agent cannot pass through.
  - `0`: Empty spaces where the agent can move.
  - `2`: The starting position of the box.
  - `3`: The target location (goal) for the box.
- **Agent Movement**: The agent can navigate the grid by moving UP, DOWN, LEFT, or RIGHT.
- **Objective**: Successfully push the box to the goal location.

## Implemented Methods

### 1. Value Iteration (Dynamic Programming)

This method uses **Value Iteration** to derive the optimal strategy for the agent to achieve its goal.

#### Key Concepts:
- **Rewards**:
  - `-1` for each step the agent takes.
  - `+100` for successfully moving the box to the goal.
  - `-100` for invalid actions, such as pushing the box into an obstacle or attempting to move into a blocked space.
- **State Transitions**: Govern how the environment changes based on the agent’s actions.
- **Optimal Policy**: Obtained by iteratively applying the Bellman equation to estimate the value function across all states.

### 2. Monte Carlo (First Visit)

The second approach uses **Monte Carlo First-Visit** to learn an optimal policy from episodic experiences.

#### Key Features:
- **Exploration vs. Exploitation**: Utilizes an **epsilon-greedy** strategy to ensure the agent explores sufficiently while also exploiting known good actions.
- **Loop Detection**: The agent identifies and avoids repeating previous states to escape loops in its decision-making process.
- **Learning from Episodes**: Updates policies based on the returns from complete episodes, using the first-visit method.
- **Challenges**: May result in suboptimal policies due to limited exploration, especially in environments with large state spaces or frequent loops.




