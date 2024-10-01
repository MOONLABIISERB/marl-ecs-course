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

### Observations
![image](https://github.com/user-attachments/assets/52494c70-eb1c-4ab4-bc36-dacfbf36b72c)
![image](https://github.com/user-attachments/assets/f6ecd475-7056-4153-b919-1d66eb231275)

  
