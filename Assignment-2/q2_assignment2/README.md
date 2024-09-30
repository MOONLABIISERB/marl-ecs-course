# Question 2

## Policy Evaluation

The table below presents the average performance of the two policies — Dynamic Programming and Monte Carlo — after training. Each policy is evaluated on the basis of two metrics: average reward and average steps.

| Policy                | Average Reward | Average Steps |
|-----------------------|----------------|---------------|
| **Dynamic Programming**   | -100.00        | 100.00        |
| **Monte Carlo**           | -77.10         | 79.52         |

### Key Observations
1. **Dynamic Programming (DP)**:
    - The **average reward** for Dynamic Programming is **-100.00**, which is indicative of consistently poor performance. 
    - The **average steps** being exactly **100** suggest that the policy is taking the maximum allowed steps in almost every episode. This indicates that the DP-based policy fails to guide the agent efficiently within the given step limit.
    - These results point to the fact that DP struggles to navigate effectively in this environment, potentially because of its inability to capture the complexity of the state space efficiently within the provided episode length.

2. **Monte Carlo (MC)**:
    - The Monte Carlo approach achieved a significantly better **average reward** of **-77.10**, demonstrating its superior performance in handling the environment's challenges.
    - The **average steps** are **79.52**, which suggests that the Monte Carlo policy is able to find solutions in fewer steps on average compared to the DP policy. 
    - This indicates that Monte Carlo is better at generalizing across states and learning an optimal path with fewer steps, even in environments with complex dynamics.

---

## Performance Comparison

The following table provides a detailed comparison of the two methods based on different metrics such as training time, average reward, and average steps.

| Metric                | Dynamic Programming | Monte Carlo |
|-----------------------|---------------------|-------------|
| **Training Time (s)**     | 23.97               | 9.04        |
| **Average Reward**        | -100.00             | -77.10      |
| **Average Steps**         | 100.00              | 79.52       |

### Detailed Analysis

1. **Training Time**:
    - **Dynamic Programming** took significantly longer to train, with a time of **23.97 seconds**, compared to just **9.04 seconds** for the Monte Carlo method. The increased time can be attributed to the fact that DP requires iterating over all possible states and actions, which becomes computationally expensive in environments with large or complex state spaces like Sokoban.
    - **Monte Carlo**, on the other hand, focuses on learning from episodes sampled through exploration, making it more efficient in terms of computation. Since it doesn’t have to process every state-action pair in the state space, it can converge faster, especially when combined with exploration-exploitation mechanisms like epsilon-greedy.

2. **Average Reward**:
    - The **average reward** of **-100.00** for Dynamic Programming suggests that this policy consistently failed to achieve positive outcomes, likely indicating that the agent is stuck in suboptimal states throughout the episode.
    - Monte Carlo’s average reward of **-77.10** shows that it learned a better policy, capable of navigating the environment more effectively and securing better outcomes. While still negative, the higher reward implies that the agent using MC encountered fewer penalties, perhaps due to shorter paths or better positioning strategies.

3. **Average Steps**:
    - Dynamic Programming used the maximum **100 steps** per episode, indicating that it was unable to find efficient solutions within the time constraints.
    - Monte Carlo averaged **79.52 steps**, indicating a much more efficient policy that could guide the agent to a solution in fewer steps on average. The ability to reduce the step count per episode is crucial for solving complex environments like Sokoban, where the optimal sequence of actions is often non-obvious.

---

## Conclusion and Thoughts

The results clearly show that **Monte Carlo** outperforms **Dynamic Programming** in this specific environment. While Dynamic Programming is an exhaustive search method and can be theoretically optimal, it faces practical challenges in complex environments like Sokoban, where the number of possible states and transitions can quickly become intractable.

### Why Monte Carlo Performed Better
- **Exploration vs. Exploitation**: Monte Carlo’s use of epsilon-greedy exploration allows it to sample episodes and learn from them, without requiring full knowledge of the environment's transition dynamics. This helps it generalize better across complex environments.
- **Scalability**: In environments with large state spaces, DP struggles because it has to compute values for every possible state-action pair. Monte Carlo, however, only updates values for the states visited in each episode, making it more scalable and efficient.
- **Flexibility**: Monte Carlo’s ability to gather experience through episodes makes it more adaptable to environments where it's difficult to model all possible transitions and outcomes, such as Sokoban.

### Potential Improvements
- **Dynamic Programming Limitations**: To improve the performance of DP, techniques like state-space pruning or hierarchical decomposition could be employed, which would help reduce the computational overhead.
- **Monte Carlo Exploration**: Monte Carlo could potentially benefit from an adaptive exploration strategy, where the epsilon value is adjusted dynamically based on the agent’s performance.

Overall, this experiment demonstrates that while Dynamic Programming is a robust algorithm, it is not always well-suited for environments with vast state spaces like Sokoban. Monte Carlo, with its ability to learn from sampled episodes, shows greater promise in such settings, providing more efficient policies and faster training times.
