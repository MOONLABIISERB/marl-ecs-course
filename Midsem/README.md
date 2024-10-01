# mid-sem
# **Q-learning on modified TSP** 

***Depedencies***
```bash
gymnasium==0.29.1
numpy==2.1.0
matplotlib==3.9.2
```
***Epsilon-greedy Q-learning***

- The travelling salesman is required to visit n places with least amount of distance travelled, repeated visits are penalized. Profits are also given as a function of last distance covered.
- Epsilon is the probability with which the best action is chosen. Starts at 0.55 with 0.5 increment every 1000 episodes eventually reaching value of 1 at which point only optimal actions are being taken.
- Learning rate set to 0.05 throughout.
- The returns improved with time and eventually converged as seen in the plot.
- Larger values of learning rate introduce noisy updates slowing down time to convergence(may or may not be optimal).
- Changing epsilon faster will give fewer opportunities for agent to explore.

![Figure_1](https://github.com/user-attachments/assets/5cb04992-6c33-4ee2-8abd-791936a9d6fe)