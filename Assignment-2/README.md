# QUestion 1

This project implements a Dynamic Programming (DP) solver. The following results were obtained by running the solver multiple times:

### DP solver

Training DP solver...
Episode 0 completed
Episode 100 completed
Episode 200 completed
Episode 300 completed
Episode 400 completed
Episode 500 completed
Episode 600 completed
Episode 700 completed
Episode 800 completed
Episode 900 completed
Optimal policy: [35 25  7 31 29  8 95 50 38 14 26 20 43  8 87 33 16 58 71 19 31 26 47 61
  0 39 43  4 83 50 41 50 14 59 90  2 78 69 24 13 94 55 48 52 54 34 84 24
 84 69 48 89 95 11 84 64 47 48 80 13 68 61 92 14 95 29 33 34  1  3  4 52
  2 25 99 91  4 64 65 43  6 39 22 22 61 11 82 79 39 62 51  6 26 92 36 68
 51 37  7 54]
Total reward using optimal policy: -930137.1345838553

### MC solver

Training MC solver...
Episode 0 completed
Episode 100 completed
Episode 200 completed
Episode 300 completed
Episode 400 completed
Episode 500 completed
Episode 600 completed
Episode 700 completed
Episode 800 completed
Episode 900 completed
Optimal policy: {0: 41, 24: 26, 39: 24, 33: 31, 1: 40, 28: 48, 27: 12, 34: 35, 25: 20, 38: 8, 2: 27, 29: 38, 3: 15, 32: 46, 42: 44, 41: 29, 44: 45, 8: 32, 19: 43, 4: 26, 13: 43, 5: 27, 21: 16, 47: 39, 14: 36, 11: 31, 6: 28, 46: 15, 17: 42, 7: 10, 36: 12, 40: 7, 9: 25, 23: 1, 45: 6, 10: 47, 16: 7, 12: 17, 31: 21, 30: 11, 20: 44, 15: 34, 26: 36, 37: 11, 18: 46, 35: 23, 22: 43, 49: 8, 43: 9, 48: 30}
Total reward using optimal policy: -180464.5052151436

Observation:

Dynamic Programming (DP) methods, such as Value Iteration and Policy Iteration, leverage the Bellman equation to iteratively compute the value of each state or state-action pair. DP relies on having a complete and accurate model of the environment, including known transition probabilities and rewards. This approach is highly efficient for problems with a well-defined, finite state space, often converging quickly because it systematically updates all states based on the structure of the problem. However, DP can be computationally intensive as the state space grows and is less effective in environments where the transition dynamics are unknown or highly complex.

Monte Carlo (MC) methods, on the other hand, estimate the value of states or actions by averaging the returns from sampled episodes, making them suitable for environments where the model is unknown or hard to define. Unlike DP, MC doesn't need a complete model, relying instead on experiences gathered during episodes. While it can be slower to converge due to the randomness of sampling, MC is more robust in complex or uncertain environments. The method's performance can vary depending on the exploration strategy used (e.g., epsilon-greedy), and its reliance on sampling can introduce variability in the results, particularly when using every-visit versus first-visit updates, which impacts learning speed and stability.

##Question 2

###Output:

![fig_game](https://github.com/user-attachments/assets/32c4927c-1327-4920-ae8c-b8dc46e05380)

output after 
```bash
python Assignment-2/q2/main.py
```
the observation about the policy are similar to the first question.

