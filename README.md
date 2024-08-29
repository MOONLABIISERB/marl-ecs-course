# Assignment 1

## Overview

This Repo contains an analysis of a finite Markov Decision Process (MDP), focusing on determining the optimal policy using both Value Iteration and Policy Iteration algorithms. 

# Question 1
The MDP is designed to model a scenario involving three locations: Hostel, Academic Building, and Canteen. The objective is to determine the optimal actions (policies) to maximize the expected rewards at each state.
### States
- **Hostel**
- **Academic Building**
- **Canteen**

### Actions
- **Attend Classes**
- **Eat Food**

### Transition Probabilities and Rewards

| **From State**      | **Action**       | **To State**          | **Transition Probability** | **Reward** |
|---------------------|------------------|-----------------------|----------------------------|------------|
| Hostel              | Attend Classes   | Academic Building      | 0.5                        | +3         |
| Hostel              | Attend Classes   | Hostel                 | 0.5                        | -1         |
| Hostel              | Eat Food         | Canteen                | 1.0                        | +1         |
| Academic Building   | Attend Classes   | Academic Building      | 0.7                        | +3         |
| Academic Building   | Attend Classes   | Hostel                 | 0.3                        | -1         |
| Academic Building   | Eat Food         | Canteen                | 0.8                        | +1         |
| Academic Building   | Eat Food         | Academic Building      | 0.2                        | +3         |
| Canteen             | Attend Classes   | Academic Building      | 0.6                        | +3         |
| Canteen             | Attend Classes   | Hostel                 | 0.3                        | -1         |
| Canteen             | Attend Classes   | Canteen                | 0.1                        | +1         |
| Canteen             | Eat Food         | Canteen                | 1.0                        | 1          |

### Diagram
![WhatsApp Image 2024-08-28 at 22 20 19](https://github.com/user-attachments/assets/0c712dc8-7935-45ab-93c6-0a7ff3bbbc5c)



## Value Iteration

The Value Iteration algorithm was applied to the MDP, resulting in the following optimal state values:

- **V(Hostel)** = 16.0556977
- **V(Academic_Building)** = 21.84597336
- **V(Canteen)** = 18.82616452

### Convergence

The algorithm converged at γ (gamma) = 0.9.

## Policy Iteration

The Policy Iteration algorithm was also applied, yielding the following optimal policy:

- **π(Hostel)** = Attend Classes
- **π(Academic_Building)** = Attend Classes
- **π(Canteen)** = Attend Classes

### Convergence

Similar to Value Iteration, the algorithm converged at γ (gamma) = 0.9.

## Conclusion

Both Value Iteration and Policy Iteration methods identified that the optimal policy at each state (Hostel, Academic Building, and Canteen) is to **Attend Classes**. The MDP model effectively demonstrates the process of finding the optimal policy in a controlled environment.

# Question 2
## Value Iteration
The Results obtain from value iteration as quiver diagram:
Where ever the value of a state is zero, there is a blockade or obstacle/wall in the gridworld environment
The algorithm converged at γ (gamma) = 0.9.
![image](https://github.com/user-attachments/assets/6ff6b984-4aa0-4d04-a914-1e73a51b2716)

## Policy Iteration
The Results obtain from Policy iteration as quiver diagram:
Where ever the value of a state is zero, there is a blockade or obstacle/wall in the gridworld environment
The algorithm converged at γ (gamma) = 0.9.

The action set is (UP, DOWN, RIGHT, LEFT) - also represented by (U,D,R,L) in the quiver plot
![image](https://github.com/user-attachments/assets/d494977d-494c-4f5c-b887-bd0c91fe2a31)


## Conclusion

Both Value Iteration and Policy Iteration methods were used to find out optimal policy at each state of the gridworld and help navigate the robot to it destination. The Results of both value iterations and Policy iterations were similar and gave optimal policy and it converged at gamma = 0.9. It was visualized using quiver plot.
The MDP model effectively demonstrates the process of finding the optimal policy in a controlled environment.
