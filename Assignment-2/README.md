# Assignment 2
This README.md file contains the analysis for the questions given in the assignment. 
All relevant outputs regarding the questions are present in the code itself.

---

Dynamic Programming (DP) methods like Value Iteration and Policy Iteration provide consistent outputs because it uses deterministic updates based on the Bellman equation, relying on a complete and known model of the environment, which stabilizes convergence to optimal values. Thus, they're best suited for simpler problems where everything about the environment is known and not too complex.

In contrast, Monte Carlo (MC) methods produce varying results due to their reliance on random sampling from episodes, which introduces variability in returns and Q-value updates. The stochastic nature of MC, combined with exploration strategies like epsilon-greedy, means different actions can be taken in different runs, leading to fluctuations in learned values, especially when the number of episodes is low.

Comparing first-visit and every-visit Monte Carlo methods, the primary distinction lies in how they update the value estimates for state-action pairs. The first-visit method updates the Q-values only for the first occurrence of a state-action pair within each episode, which can lead to a more stable estimate by reducing variance but may require more episodes to accurately reflect the true value of frequently visited pairs. In contrast, the every-visit method updates the Q-values for every occurrence of a state-action pair within an episode, allowing for more immediate learning from all experiences but potentially introducing higher variance due to multiple updates from the same episodes. As a result, every-visit can converge faster in some scenarios, while first-visit may provide more stable estimates over time, making the choice between them dependent on the specific characteristics of the problem and the desired balance between bias and variance in learning.

## Q1 - Travelling Salesman Problem (TSP):

Dynamic Programming is particularly well-suited to smaller-scale TSP problems because the state space is manageable. However, it’s important to note that as the number of cities increases, the time complexity of DP grows exponentially, making it less practical for larger problems.
While Monte Carlo methods offer more scalability in theory, they struggled with this problem due to the large state space and the difficulty of exploration. Epsilon-Greedy did help mitigate this by introducing exploration, but it wasn’t sufficient to find the best routes.

## Q2 - Sokoban Puzzle:

The Sokoban puzzle for this assignment has a small model of the environment (i.e. the transition probabilities and reward functions). It has deterministic and fully observable transitions. 
Due to the reasons discussed above and how the code performed, Dynamic Programming (like Value Iteration) could converge faster to an optimal policy.
Dynamic Programming is efficient when you have a small, well-defined, and fully-known environment, but it struggles with large state spaces.
Monte Carlo is better suited for large, complex environments or where you need to explore and learn from interaction, making it more practical for solving Sokoban puzzles in most real-world implementations.
