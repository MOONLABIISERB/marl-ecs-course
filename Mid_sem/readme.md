# Using SARSA for Travelling Salesman Problem

## Results

![download (12)](https://github.com/user-attachments/assets/8d316257-eb95-420d-a6b5-de21a59daedd)





## Discussion
SARSA is an example of on-policy. In On-policy the behavioural policy and target policy are the same.


Here action update for next state is dependent on the policy defined earlier ,in my case epsilon greedy.
As the space is large for the given TSP problem.
There are fluctuations in rewards after each episode because of large state space of TSP problem and the exploration nature of SARSA results in huge fluctuations as it explores many suboptimal paths as well.

## How to run the code:

Open in Colab -> (On ribbon) Runtime -> Run All
