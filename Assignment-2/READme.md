# Question 1


# Comparison of Dynamic Programming (DP) and Monte Carlo (MC) Solvers

## Comparison

| Metric             | Dynamic Programming (DP) | Monte Carlo (MC) |
|--------------------|--------------------------|------------------|
| **Training Time**  | 106.92s                  | 82.26s           |
| **Average Reward** | -100.00                 | -84.22           |
| **Average Steps**  | 100.00                  | 85.87            |

---

## Key Differences

### **Training Time**:
- DP took significantly more time (**106.92s**) compared to MC (**82.26s**).
- This is expected as DP requires iterative updates to all states and actions, while MC only updates based on the episodes it generates.

### **Performance (Rewards and Steps)**:
- Monte Carlo performed better with:
  - A higher average reward (**-84.22**) compared to DP (**-100.00**).
  - Fewer steps (**85.87**) than DP (**100.00**).
- MC's better performance can be attributed to its ability to learn directly from the episodes, leading to better exploration of the state space. DP, on the other hand, relies on exhaustive updates across all states, which can be limiting in exploration-heavy environments.

### **Efficiency**:
- DP is more computationally expensive due to its exhaustive computation over all states and actions, making it less efficient in larger environments.
- MC is more efficient in terms of training time, as it can perform updates based on sampled episodes, making it more suited for environments with larger state spaces or unknown dynamics.

---

## Conclusion

Monte Carlo methods are more efficient in terms of training time and tend to perform better in environments like Sokoban where exploration is crucial. 
Dynamic Programming, while more computationally expensive, offers a more deterministic approach that can be useful in scenarios with smaller or fully observable environments.
TQDm used with the help of AI to make the output of the code more interactive.

# Question 2

# Results: Comparison of Value Iteration and Monte Carlo Methods

## Value Iteration Solution:
- **Policy**: 0
- **Value Table**: [0. 0. 0. 0. 0. 0.]

### Observations:
- Value Iteration provides a deterministic policy, but the value table indicates no meaningful results. 
- This could be due to convergence issues or parameter settings in the environment.

---

## Monte Carlo Solution:
- **Policy**: [0, 2, 2, 3, 1, 5]

### Observations:
- The Monte Carlo policy is more varied, indicating exploratory behavior. 
- However, the policy may still show room for improvement depending on the environment and exploration strategy.

---

## Key Differences

### **Policy**:
- **Value Iteration**: Fixed, deterministic policy (e.g., visiting targets in order). 
- **Monte Carlo**: More exploratory, with varied actions that adapt to the state-space.

### **Value Function**:
- **Value Iteration**: Zero value table suggests convergence issues, possibly due to poor initialization or small environment size.
- **Monte Carlo**: The diverse policy suggests better exploration of the state-space, but further tuning may enhance results.

### **Exploration**:
- **Value Iteration**: Deterministic and suitable for small problems.
- **Monte Carlo**: Stochastic, making it more adaptable for larger problems but requiring sufficient exploration.

### **Scalability**:
- **Value Iteration**: Efficient for small state-spaces but computationally expensive for larger ones.
- **Monte Carlo**: Better suited for larger state-spaces, leveraging sampled episodes for learning.

### **Convergence Rate**:
- **Value Iteration**: Typically fast but struggles with poorly defined problems or large environments.
- **Monte Carlo**: Slower and heavily dependent on the number of episodes and exploration strategy.

---

## Conclusion

- **Value Iteration**: Best for small, well-defined problems where deterministic solutions are preferred.
- **Monte Carlo**: More flexible and scalable, making it a better choice for larger environments or problems requiring exploratory behavior.

---

## Recommendations

- For smaller problems, fine-tune the parameters of Value Iteration to achieve better convergence.
- For larger problems, enhance the Monte Carlo exploration strategy (e.g., by adjusting epsilon or increasing the number of episodes) to achieve optimal solutions.
