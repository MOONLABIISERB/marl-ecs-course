# Travelling Salesman Problem (TSP) using Reinforcement Learning

## Objective

The goal of the Travelling Salesman Problem (TSP) is for an agent to start at a specific point and visit 50 different targets, while minimizing the total travel cost. 

## Instructions

The TSP code is located in the `marl-ecs-course` repository, under the `Assignment 2` folder. You are required to solve the problem using two approaches:

- **Dynamic Programming (DP)**: Use either value iteration or policy iteration to find the best solution.
  
- **Monte Carlo (MC)**: Solve the TSP using the Monte Carlo method with exploring starts. Compare both the first-visit and every-visit methods.

## Results

### Performance Comparison

| Solver                          | Average Return             |
|----------------------------------|---------------------------|
| Dynamic Programming              | -59.151682476791684       |
| Monte Carlo                      | -64.28124298113774        |
| Monte Carlo Epsilon-Greedy       | -67.66526335912042        |

### Conclusion

Here’s a summary of the results:

- **Dynamic Programming**: Performed the best, giving the shortest path with the highest average return.
- **Monte Carlo Methods**: Did a bit worse but still provided reasonable results.
- **Epsilon-Greedy Exploration**: Had the worst result, showing that the extra exploration didn’t significantly improve the solution in this case.
