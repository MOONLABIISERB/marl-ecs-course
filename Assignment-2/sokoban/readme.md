### Evaluating Policies

| Policy                | Average Reward | Average Steps |
|-----------------------|----------------|---------------|
| Dynamic Programming   | -100.00        | 100.00        |
| Monte Carlo           | -77.10         | 79.52         |

### Comparison

| Metric                | Dynamic Programming | Monte Carlo |
|-----------------------|---------------------|-------------|
| Training Time (s)     | 23.97               | 9.04        |
| Average Reward        | -100.00             | -77.10      |
| Average Steps         | 100.00              | 79.52       |


### Thoughts
This time, MC performed much better. This is probably due to the fact situation is too complex for DP to explore all scenarios within the episode length