# Assignment 2: Q1 - Sokoban

### Comparison of Sokoban metrics for DP and MC methods

| Metric         | Dynamic Programming | Monte Carlo |
|----------------|---------------------|-------------|
| Training Time  | 20.38s              | 7.48s       |
| Average Reward | -100.00             | -87.49      |
| Average Steps  | 100.00              | 88.81       |

### Explanation

- Dynamic Programming (DP) requires full knowledge of all outcomes of the environment model.
- The given Sokoban environment is too complex for DP to converge fast.
- Monte-Carlo (MC) method works using random sampling of episodes in the environment and their returns, relying on the **Law of Large Numbers** to get a good estimate of the action-value function.
- This is why MC converges quicker than DP.
