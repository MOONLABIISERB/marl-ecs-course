# Travelling Salesman

In the first problem, the Markov Decision Process (MDP) is defined as follows:

1. **States**: Each state is represented as a 2-tuple, where the first element is the current node and the second is a list of previously visited nodes.
2. **Actions**: The available actions consist of all possible targets, i.e., (0, 1, 2, 3, 4, 5) for an example with 6 targets.
3. **Start State**: The initial state is `(0, ())`, where node 0 is the start, and the list of visited nodes is empty.
4. **Deterministic Transitions**: The transition probability is always 1 for any state-action pair since the environment is deterministic. The agent always transitions to a specific state given the current state and action.
5. **Reward Structure**: 
   - A negative reward equal to the distance between the current state and the next state is assigned when moving to an unvisited node.
   - A large penalty is given if the agent revisits an already visited node.

# Dynamic Programming vs. Monte Carlo in Reinforcement Learning

This section highlights the main differences between **Dynamic Programming (DP)** and **Monte Carlo (MC)** methods in reinforcement learning:

## 1. Requirements

- **DP**: Needs a complete model of the environment, including transition probabilities `P(s'|s, a)` and the reward function `R(s, a, s')`.
- **MC**: Does not need an explicit environment model. It learns directly from episodes that are sampled (sequences of state-action-reward pairs).

## 2. State-Action Value Updates

### DP (e.g., Value Iteration)
- Updates state values using the Bellman equation.
- Iterates over all states and actions, adjusting values based on expected rewards and the value of future states.

### MC (e.g., First-Visit Monte Carlo)
- Estimates state-action values based on returns from sampled episodes.
- It does not require knowledge of future states ahead of time; instead, it learns from actual experiences.



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
