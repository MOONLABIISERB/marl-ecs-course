# MARL
Assignments 1 for course MARRL
- To evaluate the questions of the assignment-1, please open the file named `qX_assignment1.py`. For example, `X = 1` for assignment #1 and so on.

## Question 1:

### States
- Hostel
- AB
- Canteen

### Actions
- (AC) Attend Classes
- (EF) Eat Food
  
#RESULTS
### Transition Probabilities and Rewards

| From State       | Action         | To State          | Transition Probability | Reward |
|------------------|----------------|-------------------|------------------------|--------|
| Hostel           | AC             | AB                | 0.5                    | +3     |
| Hostel           | AC             | Hostel            | 0.5                    | -1     |
| Hostel           | EF             | Canteen           | 1.0                    | +1     |
| AB               | AC             | AB                | 0.7                    | +3     |
| AB               | AC             | Hostel            | 0.3                    | -1     |
| AB               | EF             | Canteen           | 0.8                    | +1     |
| AB               | EF             | AB                | 0.2                    | +3     |
| Canteen          | AC             | AB                | 0.6                    | +3     |
| Canteen          | AC             | Hostel            | 0.3                    | -1     |
| Canteen          | AC             | Canteen           | 0.1                    | +1     |
| Canteen          | EF             | Canteen           | 1.0                    | +1     |

Here γ (gamma) = 0.9.

## Value Iteration
The Value Iteration algorithm was applied to the MDP, resulting in the following optimal state values:

- **V(Hostel)** = 18.95
- **V(AB)** = 20.94
- **V(Canteen)** = 19.81

## Policy Iteration
The Policy Iteration algorithm was also applied, yielding the following optimal policy:

- **π(Hostel)** = AC
- **π(AB)** = AC
- **π(Canteen)** = AC

'Report.pdf' contains the transition probabilities and rewards table and diagram for the question 1 of the assignment 1.
Both Value Iteration and Policy Iteration methods yielded the same optimal policy and very similar optimal values for the states. The policy suggests that the student should focus on attending classes in all states to maximize long-term rewards. This is intuitive, as attending classes in the Academic Building provides the highest immediate reward (+3), and the transitions between states are structured to favor moving towards the Academic Building. The minor differences in value results are due to the iterative nature of the algorithms, but they both converge to the same optimal policy.

## Question 2
![Policy iteration](https://github.com/user-attachments/assets/f2761d59-8e85-42fb-b67f-32e0924596ea)
![Value iteration](https://github.com/user-attachments/assets/85269f24-9a20-4c34-9471-a30714058657)
