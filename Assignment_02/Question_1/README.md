
# Travelling Salesman Problem (TSP) using Reinforcement Learning

## Objective

The goal of the Travelling Salesman Problem (TSP) is for an agent to start at a specific point and visit 50 different targets, while minimizing the total travel cost.

## Approaches

This project implements two reinforcement learning-based approaches to solve the TSP:

### 1. Dynamic Programming (DP)
- **Method**: Value Iteration
- Uses a cost table to store intermediate results and calculates the optimal policy by minimizing the total journey cost.

### 2. Monte Carlo (MC)
- **Methods**: 
  - First-Visit Monte Carlo
  - Every-Visit Monte Carlo
- Both methods simulate episodes to learn optimal policies using sampled returns.

## Implementation Details

- **Dynamic Programming**: Implements a solver that precomputes all possible subsets of cities and their costs to find the optimal path.
- **Monte Carlo**: Includes two approaches:
  1. First-Visit: Updates state-action values only on the first occurrence in an episode.
  2. Every-Visit: Updates state-action values on every occurrence in an episode.

## Results

### Performance Comparison

| Solver                          | Average Return (Lower is Better) |
|----------------------------------|-----------------------------------|
| Dynamic Programming              | -59.15                           |
| Monte Carlo (First-Visit)        | -61.84                         |
| Monte Carlo (Every-Visit)        | -67.11                         |

### Observations

1. **Dynamic Programming** consistently produces the best results as it explores all possibilities deterministically.
2. **Monte Carlo Methods** show variability due to their reliance on random sampling but still perform reasonably well.
3. Between the Monte Carlo approaches, Every-Visit tends to converge faster due to more frequent updates.

## Instructions

- Ensure you have Python 3.x and the required dependencies installed.
- Run the `marl_assi2_tsp_modified.ipynb` notebook to train and evaluate both methods.

## Conclusion

Dynamic Programming is more computationally intensive but yields the best solutions. Monte Carlo methods, while less optimal, provide a flexible and scalable alternative for larger problems.

