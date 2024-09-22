# Assignment 2: Q1 - Travelling Salesman Problem

## Performance Comparison

| Method                        | Average Return     |
|-------------------------------|---------------------|
| Dynamic Programming           | -67.69              |
| Monte Carlo (First-Visit)     | -40,013.02          |
| Monte Carlo (Every-Visit)     | -40,013.02          |
| Monte Carlo (Epsilon-Greedy)  | -10,063.16          |

## Explanation

- Dynamic Programming outperformed Monte Carlo methods significantly.
- DP found efficient routes (avg. return -67.69) vs. MC's poor results.
- Monte Carlo methods struggled with TSP's combinatorial complexity.
  - Standard MC approaches performed worst (avg. return -40,013.02).
  - Epsilon-Greedy MC showed improvement (-10,063.16) due to balanced exploration.
  - Despite improvement, Epsilon-Greedy still fell far short of DP's performance.
- DP's success likely due to exhaustive search, ideal for smaller problems.

## Conclusion

- DP should be used for simpler problems like the TSP.
- Monte-Carlo Method should be used for complex problems which have a very large state/action space.
