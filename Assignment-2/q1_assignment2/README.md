# Assignment 2
# Question1: Travelling Salesman Problem

## Performance Comparison

The table below shows the average return (cumulative distance) for various methods used to solve the Travelling Salesman Problem (TSP). The goal is to find the shortest possible route that visits each target once and returns to the starting point.

| Method                           | Average Return          |
|----------------------------------|-------------------------|
| **Dynamic Programming**              | -84.70                |
| **Monte Carlo (First-Visit)**        | -40015.35             |
| **Monte Carlo (Every-Visit)**        | -40015.35             |
| **Monte Carlo (Epsilon-Greedy)**     | -10080.52             |

### Key Observations:

1. **Dynamic Programming (DP)**:
    - The **average return** for Dynamic Programming is **-84.70**, which indicates that this method was able to find efficient routes with shorter distances.
    - DP outperformed all other methods significantly, with much smaller negative returns, implying it found near-optimal or optimal solutions to the TSP.
    - The success of DP here can be attributed to its ability to exhaustively explore all possible routes, ensuring that it identifies the shortest possible path.

2. **Monte Carlo (MC) Methods**:
    - Both **Monte Carlo (First-Visit)** and **Monte Carlo (Every-Visit)** performed poorly, yielding extremely large negative returns of **-40015.35**.
    - This suggests that the Monte Carlo methods struggled to converge to good solutions for the TSP in the current setup. These methods likely explored suboptimal routes, leading to significantly longer distances and, therefore, worse performance.
    - **Monte Carlo (Epsilon-Greedy)** showed some improvement with an average return of **-10080.52**, which is better than the other Monte Carlo approaches but still far worse than DP. The use of epsilon-greedy exploration allowed for some exploration of better routes, but overall, it still did not find efficient solutions.

---

## Detailed Analysis

1. **Dynamic Programming**:
    - **Why DP Performed Well**: Dynamic Programming systematically explores every possible route in the TSP, calculating the cost of each route and storing intermediate results to avoid redundant calculations. This allows DP to find the exact optimal solution in small to medium-sized problems like the one in this assignment. Since DP evaluates every route, it ensures that it doesn’t miss any potential shorter paths.
    - **Suitability for TSP**: DP is particularly well-suited to smaller-scale TSP problems because the state space is manageable. However, it’s important to note that as the number of cities increases, the time complexity of DP grows exponentially, making it less practical for larger problems.

2. **Monte Carlo Methods**:
    - **First-Visit and Every-Visit MC**: Both of these methods produced poor results, with nearly identical performance. The large negative returns indicate that the MC approaches struggled to learn good policies or explore efficient routes in this environment. Monte Carlo relies on sampling episodes to estimate value functions, but without enough exploration, it can get stuck in suboptimal policies, leading to poor performance.
        - **First-Visit**: This method updates the state-action pair the first time it is encountered in an episode. However, since it doesn’t guarantee extensive exploration, it might fail to explore all possible routes.
        - **Every-Visit**: Every-Visit updates the state-action pair every time it is encountered in an episode, but this can lead to inefficient updates and convergence issues in environments with large state spaces like the TSP.
    - **Epsilon-Greedy MC**: Epsilon-greedy Monte Carlo performed better, although still far worse than DP. The exploration introduced by the epsilon-greedy policy helped the agent explore different routes more effectively. However, the average return of **-10080.52** suggests that even with exploration, the MC agent struggled to consistently find optimal or even near-optimal routes.

---

## Insights and Takeaways

### Why DP Worked Better:
- **Comprehensive Search**: DP explores every possible route, ensuring it finds the best one. This exhaustive search is feasible for the current problem size, where the number of cities is small enough for DP to handle.
- **Deterministic Nature**: Unlike the stochastic Monte Carlo methods, DP is deterministic and doesn’t rely on sampling. It works by breaking down the problem into subproblems and solving each optimally, which is perfect for combinatorial optimization problems like TSP.

### Why Monte Carlo Struggled:
- **State Space Complexity**: The TSP has a vast number of possible routes, even for a relatively small number of cities. Monte Carlo methods rely on episode sampling, and without sufficient exploration, they often fail to sample all the critical parts of the state space. In this case, it appears that the Monte Carlo methods frequently got stuck in local optima or suboptimal routes, leading to poor returns.
- **Dependence on Exploration**: The performance of Monte Carlo algorithms is heavily influenced by how well they explore the state space. The poor performance of First-Visit and Every-Visit suggests that the exploration was inadequate. Epsilon-Greedy helped mitigate this by introducing exploration, but it wasn’t sufficient to find the best routes.

---

## Conclusion

In this assignment, **Dynamic Programming** emerged as the clear winner for solving the Travelling Salesman Problem, thanks to its ability to evaluate all possible routes and find the shortest path. While **Monte Carlo** methods offer more scalability in theory, they struggled with this problem due to the large state space and the difficulty of exploration. 

### Future Improvements:
- **Enhancing Monte Carlo Exploration**: To improve the performance of Monte Carlo methods, one could explore strategies like adaptive epsilon or decaying epsilon schedules, which might help the agent better explore the environment and converge to more optimal policies.
- **Scaling DP**: While DP works well for small instances of TSP, future work could focus on exploring heuristics or approximations to make DP scalable to larger problems, such as combining it with techniques like pruning or approximation algorithms.

Overall, this experiment highlights the strengths and weaknesses of different approaches to solving combinatorial optimization problems like TSP and offers insights into when each method is best applied.
