# Sokoban Puzzle

###  Run Code
```
python main.py

```


## Results:

| Metric                    | Dynamic Programming Agent | Monte Carlo Agent       |
|---------------------------|---------------------------|-------------------------|
| **Training Time**         | 19.71 seconds             | 5.14 seconds            |
| **Average Reward**        | -100.00                   | -92.59                  |
| **Average Steps**         | 100.00                    | 93.36                   |

### Comparison

- **Training Time**: The Monte Carlo Agent trained significantly faster than the Dynamic Programming Agent.
- **Average Reward**: Both agents received negative rewards, indicating room for improvement; however, the Monte Carlo Agent performed better.
- **Average Steps**: The Monte Carlo Agent used fewer steps on average to solve the puzzle compared to the Dynamic Programming Agent.
