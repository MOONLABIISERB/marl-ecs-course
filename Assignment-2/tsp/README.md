# assignment-2

## Travelling salesman problem
| Method                           | Average Return          |
|----------------------------------|-------------------------|
| Dynamic Programming              | -84.7032951572178       |
| Monte Carlo (First-Visit)        | -40015.34689341699      |
| Monte Carlo (Every-Visit)        | -40015.34689341699      |
| Monte Carlo (Epsilon-Greedy)     | -10080.523009384502     |

### What we learned:

- DP worked much better than MC for this problem.
- DP found good routes with shorter distances.
- MC methods didn't work well and gave bad results.
- MC(epsilong-greedy) was a bit better as it explored more.

### Why this migth have happened:

DP checks all possible ways to solve the problem. It's good for small problems like this.

