
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
![alt text](https://github.com/MOONLABIISERB/marl-ecs-course/blob/gavit_20114/Assignment-1/MDP.png)

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

# Question 2: Optimal Values and Policies

## a) Value Iteration

### Optimal Values

| Rows | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 0    | 0.914 | 0.923 | 0.932 | 0.923 | 0.914 | 0.904 | 0.895 | 0.886 | 0.878 |
| 1    | 0.923 | 0.932 | 0.941 | 0.000 | 0.904 | 0.895 | 0.886 | 0.878 | 0.869 |
| 2    | 0.932 | 0.941 | 0.951 | 0.000 | 0.895 | 0.886 | 0.878 | 0.869 | 0.860 |
| 3    | 0.923 | 0.000 | 0.000 | 0.000 | 0.886 | 0.878 | 0.869 | 0.860 | 0.851 |
| 4    | 0.914 | 0.904 | 0.895 | 0.886 | 0.878 | 0.869 | 0.860 | 0.851 | 0.843 |
| 5    | 0.904 | 0.895 | 0.886 | 0.877 | 0.869 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6    | 0.895 | 0.886 | 0.878 | 0.869 | 0.860 | 0.000 | 0.961 | 0.970 | 0.980 |
| 7    | 0.886 | 0.878 | 0.869 | 0.860 | 0.851 | 0.000 | 0.970 | 0.980 | 0.990 |
| 8    | 0.878 | 0.869 | 0.860 | 0.851 | 0.843 | 0.000 | 0.980 | 0.990 | 1.000 |

### Optimal Policies

| Rows | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
|------|------|------|------|------|------|------|------|------|------|
| 0    | up   | up   | up   | left | left | left | left | left | left |
| 1    | up   | up   | up   | up   | down | down | down | down | down |
| 2    | right| right| down | up   | down | down | down | down | down |
| 3    | down | up   | up   | up   | down | down | down | down | down |
| 4    | down | left | left | left | down | down | down | down | down |
| 5    | down | down | down | down | down | up   | up   | up   | up   |
| 6    | down | down | down | down | down | up   | up   | up   | up   |
| 7    | down | down | down | down | down | up   | up   | up   | up   |
| 8    | down | down | down | down | down | up   | right| right| up   |

### Quiver Plot
![alt text](https://github.com/MOONLABIISERB/marl-ecs-course/blob/gavit_20114/Assignment-1/MDP.png)


## b) Policy Iteration

### Optimal Values

| Rows | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 0    | 0.914 | 0.923 | 0.932 | 0.923 | 0.914 | 0.904 | 0.895 | 0.886 | 0.878 |
| 1    | 0.923 | 0.932 | 0.941 | 0.000 | 0.904 | 0.895 | 0.886 | 0.878 | 0.869 |
| 2    | 0.932 | 0.941 | 0.951 | 0.000 | 0.895 | 0.886 | 0.878 | 0.869 | 0.860 |
| 3    | 0.923 | 0.000 | 0.000 | 0.000 | 0.886 | 0.878 | 0.869 | 0.860 | 0.851 |
| 4    | 0.914 | 0.904 | 0.895 | 0.886 | 0.878 | 0.869 | 0.860 | 0.851 | 0.843 |
| 5    | 0.904 | 0.895 | 0.886 | 0.877 | 0.869 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6    | 0.895 | 0.886 | 0.878 | 0.869 | 0.860 | 0.000 | 0.961 | 0.970 | 0.980 |
| 7    | 0.886 | 0.878 | 0.869 | 0.860 | 0.851 | 0.000 | 0.970 | 0.980 | 0.990 |
| 8    | 0.878 | 0.869 | 0.860 | 0.851 | 0.843 | 0.000 | 0.980 | 0.990 | 1.000 |

### Optimal Policies

| Rows | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
|------|------|------|------|------|------|------|------|------|------|
| 0    | up   | up   | up   | left | left | left | left | left | left |
| 1    | up   | up   | up   | up   | down | down | down | down | down |
| 2    | right| right| down | up   | down | down | down | down | down |
| 3    | down | up   | up   | up   | down | down | down | down | down |
| 4    | down | left | left | left | down | down | down | down | down |
| 5    | down | down | down | down | down | up   | up   | up   | up   |
| 6    | down | down | down | down | down | up   | up   | up   | up   |
| 7    | down | down | down | down | down | up   | up   | up   | up   |
| 8    | down | down | down | down | down | up   | right| right| up   |

### Quiver Plot
![alt text](https://github.com/MOONLABIISERB/marl-ecs-course/blob/gavit_20114/Assignment-1/MDP.png)


