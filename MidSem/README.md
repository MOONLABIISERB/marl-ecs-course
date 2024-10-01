# MidSem
This README.md file contains the analysis for the problem given for the MidSem exam. All relevant outputs regarding the questions are present in the code itself.

---
## TD-based reinforcement learning algorithm used - Q-Learning
- Q-Learning is a model-free reinforcement learning algorithm. Is off-policy, meaning the agent learns from actions it doesn't necessarily take (via exploration).
- It learns the optimal action-selection policy for an agent by updating Q-values.
- The core update rule:
Q(s, a) ← Q(s, a) + α [r + γ max(Q(s', a')) - Q(s, a)],
where:
  - α is the learning rate (0 < α ≤ 1).
  - γ is the discount factor (0 ≤ γ ≤ 1).
  - r is the reward received after taking action a from state s.
  - max(Q(s', a')) is the maximum expected future reward from the next state s'..
- The agent balances exploration (trying new actions) and exploitation (choosing the best-known action) using strategies like epsilon-greedy.
- Converges to the optimal policy as long as every state-action pair is visited infinitely often and learning parameters are properly tuned

---
### Hyperparameters Used - 
**Environment (ModTSP)**
- num_targets: 10 (Number of targets the agent needs to visit)
- max_area: 15
- shuffle_time: 50 (Number of episodes after which profits are shuffled)
- seed: 42

**Q-Learning Agent (QAgent)**
- learning_rate: 0.1
- discount_factor: 0.99
- number of episodes: 999

### Observations -
![image](https://github.com/user-attachments/assets/52494c70-eb1c-4ab4-bc36-dacfbf36b72c)
![image](https://github.com/user-attachments/assets/f6ecd475-7056-4153-b919-1d66eb231275)

| Episode   | Reward                |
|-----------|-----------------------|
| 100 / 999 | -9845.45              |
| 200 / 999 | -9779.49              |
| 300 / 999 | 138.25                |
| 400 / 999 | 145.54                |
| 500 / 999 | 133.38                |
| 600 / 999 | 159.57                |
| 700 / 999 | 159.57                |
| 800 / 999 | 159.57                |
| 900 / 999 | 159.57                |


- Converges at reward 159.56.
- Starting around episode 600, the agent consistently collects a total reward of approximately 159.56, indicating that the agent has converged to an optimal policy.
- Converging to Positive reward means that the agent is not revisiting any target.
- Initially the agent was having highly negative rewards. This was because the agent was initially exploring and learning, sometimes revisiting targets, which incurs heavy penalties.
- There were few sporadic events resulting in negative reward which were due to revisiting targets, causing the penalty.

### Conclusion - 
Q-Learning worked better compared to other algorithms for the following reasons - 

- Q-Learning is an off-policy algorithm, meaning it learns the optimal policy independently of the agent's current behavior. This is not the case for SARSA. 
- Q-Learning is much simpler to implement and debug compared to DQN.
- Q-Learning's tabular method is more than sufficient and avoids the extra complexity of training and tuning a neural network.
- Q-Learning generally converges faster in small, discrete problems like TSP because there’s no need to tune complex neural network parameters.
