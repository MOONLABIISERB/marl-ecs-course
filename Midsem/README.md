# mid-sem
# **Q-learning on modified TSP** 

***Depedencies***
```bash
gymnasium
numpy
matplotlib
```
***Epsilon-greedy Q-learning implementation***

- The travelling salesman is required to visit n places with least amount of distance travelled, repeated visits are penalized. Profits are also given as a function of last distance covered.
- Epsilon is the probability with which the best action is chosen. Starts at 0.55 with 0.5 increment every 1000 episodes eventually reaching value of 1 at which point only optimal actions are being taken.
- Learning rate set to 0.05 throughout.
- The returns improved with time and eventually converged as seen in the plot.


![Figure_1](https://github.com/user-attachments/assets/5cb04992-6c33-4ee2-8abd-791936a9d6fe)