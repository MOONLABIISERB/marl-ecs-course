# Mid-Semester Examination

Name: Rohan Mehra  
Roll No: 21224

## Answer: Modified TSP Problem

Implementing SARSA (which used epsilon-greedy policy) to solve with decaying epsilon.

+ **Learning Rate (α):** A higher learning rate speeds up learning but may lead to instability.  
+ **Discount Factor (γ):** Increasing the discount factor encourages long-term rewards, while lowering it focuses on immediate gains.  
+ **Epsilon (ε):** High epsilon early encourages exploration; it decays over time for more exploitation.  
+ **Number of Episodes:** More episodes lead to a better-trained agent but require more computation time.

### Run 1

| Parameter                | Value                                     | 
|--------------------------|-------------------------------------------|
| Learning Rate (alpha)    | 0.01                                      | 
| Discount Factor (gamma)  | 0.99                                      | 
| Epsilon                  | 0.1                                       | 
| Epsilon Decay Rate       | epsilon = max(0.00005, epsilon * 0.9999)  | 
| Number of Episodes       | 10000                                     | 

<img src="https://github.com/user-attachments/assets/b7eef364-de0e-421a-905f-b7697966561e" alt="Graph-TSP-SARSA" width="700"/>

<img src="https://github.com/user-attachments/assets/ef0a48ad-db10-4b1b-ab0a-03ab96a5cb7b" alt="Path-TSP-SARSA" width="700"/>

### Run 2

| Parameter                | Value                                     | 
|--------------------------|-------------------------------------------|
| Learning Rate (alpha)    | 0.04                                       | 
| Discount Factor (gamma)  | 0.99                                      | 
| Epsilon                  | 0.1                                       | 
| Epsilon Decay Rate       | max(0.00005, epsilon * 0.9999)            | 
| Number of Episodes       | 80000                                     |

<img src="https://github.com/user-attachments/assets/13c0ac31-46a5-43df-bf79-6f801c882c37" width="700"/>

<img src="https://github.com/user-attachments/assets/6d562eea-f0e5-4cb9-b46e-c7c1931300f4" width="700"/>



