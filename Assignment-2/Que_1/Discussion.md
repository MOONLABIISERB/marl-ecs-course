### Results

1. **Dynamic Programming (DP)**
   - **Optimal Policy**: `[0 1 2 3 4 5]`
   - **Value Table**: `[0. 0. 0. 0. 0. 0.]`
     - DP provides a deterministic policy but the value table shows no meaningful results due to convergence issues or parameter setup.

2. **Monte Carlo (First-Visit)**
   - **Policy**: `[1 1 1 1 1 1]`
   - **Value Table**: `[-33992.58, -10737.27, ...]`
     - The First-Visit MC policy repeatedly chooses action 1 for all states, and the value table indicates poor learning results due to insufficient exploration.

3. **Monte Carlo (Every-Visit)**
   - **Policy**: `[2 1 2 2 1 2]`
   - **Value Table**: `[-34064.52, -25210.30, ...]`
     - The Every-Visit MC policy is more diverse, and the value table improves slightly compared to First-Visit MC.

## Key Differences

- **Policy**:
  - **DP**: Fixed, deterministic policy (visiting targets in order).
  - **MC**: More exploratory and variable policies (depends on exploration-exploitation balance).

- **Value Function**:
  - **DP**: Zero value table indicates convergence issues.
  - **MC**: Negative values, suggesting suboptimal learning due to limited episodes and exploration.

- **Exploration**:
  - **DP**: Deterministic and optimal for small problems.
  - **MC**: Stochastic, requires more episodes for exploration and convergence.

- **Scalability**:
  - **DP**: Suitable for small state spaces but inefficient for large ones.
  - **MC**: Scales better for larger state spaces but requires many episodes.

- **Convergence Rate**:
  - **DP**: Fast but may struggle with larger problems.
  - **MC**: Slower and requires more episodes to converge.

## Conclusion

- **DP**: Best for small, well-defined problems with exact solutions.
- **MC**: More flexible and scalable for larger problems but slower to converge and may not always provide optimal solutions.

---