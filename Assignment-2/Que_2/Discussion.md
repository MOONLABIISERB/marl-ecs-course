## Results

### Comparison

| Metric            | Dynamic Programming (DP) | Monte Carlo (MC) |
|-------------------|--------------------------|------------------|
| **Training Time**  | 40.44s                   | 21.59s           |
| **Average Reward** | -98.90                   | -80.94           |
| **Average Steps**  | 99.01                    | 82.92            |

### Key Differences

1. **Training Time**: 
   - DP took significantly more time (40.44s) compared to MC (21.59s). This is expected as DP requires iterative updates to all states and actions, while MC only updates based on the episodes it generates.
   
2. **Performance (Rewards and Steps)**:
   - **Monte Carlo** performed better with a higher average reward (-80.94) and fewer steps (82.92) compared to **Dynamic Programming**, which had a lower reward (-98.90) and more steps (99.01).
   - MC's better performance can be attributed to its ability to learn directly from the episodes, which may lead to better exploration of the state space compared to the more deterministic updates in DP.

3. **Efficiency**:
   - DP is more computationally expensive due to its exhaustive computation over all states and actions, which can be a bottleneck in larger environments.
   - MC is more efficient in terms of training time, as it can perform updates based on sampled episodes, making it more suited for environments with larger state spaces or unknown dynamics.

## Conclusion

Monte Carlo methods are more efficient in terms of training time and tend to perform better in environments like Sokoban where exploration is crucial. Dynamic Programming, while more computationally expensive, offers a more deterministic approach that can be useful in certain scenarios with smaller or fully observable environments.