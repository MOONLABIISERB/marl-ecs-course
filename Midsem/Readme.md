
# Q-Learning Agent Results

This table presents a summary of the Q-learning agent's total reward collected at various episode intervals during training.

| Episode Number | Total Reward Collected |
|----------------|------------------------|
| 0              | -79977.53               |
| 100            | 126.06                  |
| 200            | 126.06                  |
| 300            | -19789.25               |
| 400            | 124.45                  |
| 500            | 145.31                  |
| 600            | -9860.13                |
| 700            | 108.78                  |
| 800            | 115.21                  |
| 900            | -19830.14               |
| 1000           | -19863.95               |
| 1100           | 126.06                  |
| 1200           | 139.95                  |
| 1300           | 124.41                  |
| 1400           | 145.31                  |
| 1500           | 145.31                  |
| 1600           | 145.31                  |
| 1700           | 145.31                  |
| 1800           | 145.31                  |
| 1900           | 145.31                  |
| 2000           | 145.31                  |
| 3000           | 145.31                  |
| 4000           | 145.31                  |
| 5000           | 145.31                  |
| 6000           | 145.31                  |
| 7000           | 145.31                  |
| 8000           | 145.31                  |
| 9000           | 145.31                  |
| 9900           | 145.31                  |

> **Note**: The agent converged around episode 1400, consistently achieving a reward of **145.31** from then on.


## Observations:

**Convergence:** Starting around episode 1400, the agent consistently collects a total reward of approximately 145.31, indicating the agent has likely converged to an optimal policy.

**Initial High Variability:** Early episodes show significant variability in rewards, with some episodes having highly negative rewards (e.g., -79977 and -19863). This suggests that the agent was initially exploring and learning, sometimes revisiting targets, which incurs heavy penalties.

**Stable Performance:** After convergence, the rewards remain constant at 145.31, suggesting that the agent has learned a stable policy and is performing consistently well by avoiding penalties.

**Negative Rewards in Some Episodes:** There are sporadic episodes with significant negative rewards (e.g., Episode 600: -9860). This could be due to revisiting targets in those particular episodes, causing the penalty.



## Cumulative Reward per Episode and Average Loss:

![cr](https://github.com/user-attachments/assets/b9272778-3c12-4e48-aa15-d6f49716e54d)









