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




