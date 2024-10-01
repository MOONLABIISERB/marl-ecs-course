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

<img src="https://github.com/user-attachments/assets/619c2ffd-79be-49dd-821d-b1bd958ae6a2" alt="Graph-TSP-SARSA" width="700"/>

<img src="https://github.com/user-attachments/assets/6c9a8d0c-17f8-4319-90cc-8bfd894f1f89" alt="Graph-TSP-SARSA" width="700"/>

### Run 2

| Parameter                | Value                                     | 
|--------------------------|-------------------------------------------|
| Learning Rate (alpha)    | 0.1                                       | 
| Discount Factor (gamma)  | 0.95                                      | 
| Epsilon                  | 0.1                                       | 
| Epsilon Decay Rate       | epsilon = max(0.01, epsilon * 0.995)      | 
| Number of Episodes       | 3000000                                   |

<img src="https://github.com/user-attachments/assets/6ad2165c-1590-4466-b335-c94658d63f27" alt="Graph-TSP-SARSA" width="700"/>



