# MidSem
This README.md file contains the analysis for the problem given for the MidSem exam. All relevant outputs regarding the questions are present in the code itself.

---
## TD-based reinforcement learning algorithm used - Q-Learning
Q-Learning is a model-free reinforcement learning algorithm.
It learns the optimal action-selection policy for an agent by updating Q-values.
Q-values represent the expected future rewards for action a in state s.
The core update rule:
Q(s, a) ← Q(s, a) + α [r + γ max(Q(s', a')) - Q(s, a)],
where:
α is the learning rate (0 < α ≤ 1).
γ is the discount factor (0 ≤ γ ≤ 1).
r is the reward received after taking action a from state s.
max(Q(s', a')) is the maximum expected future reward from the next state s'.
Q-Learning is off-policy, meaning the agent learns from actions it doesn't necessarily take (via exploration).
The agent balances exploration (trying new actions) and exploitation (choosing the best-known action) using strategies like epsilon-greedy.
Converges to the optimal policy as long as every state-action pair is visited infinitely often and learning parameters are properly tuned
