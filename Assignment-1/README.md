
# Assignment 1: Markov Decision Process (MDP) Analysis

## Question 1

### States
- Hostel
- Academic Building
- Canteen

### Actions
- Attend Classes
- Eat Food
  
#RESULTS
### Transition Probabilities and Rewards

| From State       | Action         | To State          | Transition Probability | Reward |
|------------------|----------------|-------------------|------------------------|--------|
| Hostel           | Attend Classes | Academic Building | 0.5                    | +3     |
| Hostel           | Attend Classes | Hostel            | 0.5                    | -1     |
| Hostel           | Eat Food       | Canteen           | 1.0                    | +1     |
| Academic Building| Attend Classes | Academic Building | 0.7                    | +3     |
| Academic Building| Attend Classes | Hostel            | 0.3                    | -1     |
| Academic Building| Eat Food       | Canteen           | 0.8                    | +1     |
| Academic Building| Eat Food       | Academic Building | 0.2                    | +3     |
| Canteen          | Attend Classes | Academic Building | 0.6                    | +3     |
| Canteen          | Attend Classes | Hostel            | 0.3                    | -1     |
| Canteen          | Attend Classes | Canteen           | 0.1                    | +1     |
| Canteen          | Eat Food       | Canteen           | 1.0                    | +1     |

### Diagram


### Convergence
The algorithm converged with a discount factor of γ (gamma) = 0.9.

## Value Iteration
The Value Iteration algorithm was applied to the MDP, resulting in the following optimal state values:

- **V(Hostel)** = 18.95
- **V(Academic Building)** = 20.94
- **V(Canteen)** = 19.81

## Policy Iteration
The Policy Iteration algorithm was also applied, yielding the following optimal policy:

- **π(Hostel)** = Attend Classes
- **π(Academic Building)** = Attend Classes
- **π(Canteen)** = Attend Classes


