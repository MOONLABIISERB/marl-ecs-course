# MIDSEM - modified TSP

This repository implements the Traveling Salesman Problem (TSP) as a reinforcement learning task using the SARSA algorithm within a custom Gymnasium environment. The agent's objective is to maximize profits by visiting targets (cities), while considering that profits decay over time based on travel distance.

---
1. **Clone the repository**:
    ```bash
    git clone https://github.com/MOONLABIISERB/marl-ecs-course.git
    cd marl-ecs-course
    ```

2. **Switch to the branch** `Ankur_21043`:
    ```bash
    git checkout Ankur_21043
    ```

3. **Install dependencies**:
    Make sure Python 3.7+ is installed, then install the required dependencies using `pip`:

    ```bash
    pip install gymnasium numpy matplotlib
    ```

4. **Run the SARSA algorithm to reproduce results**:
    Execute the main script to run the SARSA algorithm:

    ```bash
    python MARL_midsem.py
    ```

This will train the agent using the SARSA algorithm and produce the cumulative rewards plot and optimal policy visualization.
---
## Implementation Details

The environment `ModTSP` is a custom Gymnasium environment where:
- **Targets** are randomly placed within a defined square area.
- The agent must visit all targets, with each target having an associated profit that decreases the longer the agent takes to reach it.

Key features of the implementation:
- **State Space**: The environment state is represented as a combination of:
  1. The agent's current location.
  2. Flags indicating whether each target has been visited.
  3. The current profits available at each target.
  4. The distances from the agent's current location to all other targets.
  5. The (x, y) coordinates of all targets.
  
- **Action Space**: The agent selects its next target from a discrete action space corresponding to the available unvisited targets.

### Hyperparameters for SARSA
The SARSA algorithm in this implementation uses the following hyperparameters:
- **Episodes**: `80,000` (total number of training episodes).
- **Learning rate (alpha)**: `0.00001`.
- **Discount factor (gamma)**: `0.99`.
- **Epsilon**: `0.0001` (controls exploration rate in the epsilon-greedy policy).
- **Shuffle time**: `10` (the episode after which the target profits are shuffled).

---

## SARSA Explanation

The **SARSA** (State-Action-Reward-State-Action) algorithm is an on-policy reinforcement learning method. It works by estimating the action-value function $`Q(s, a)`$ for each state-action pair based on the interactions with the environment. SARSA updates the Q-values using the following formula:

$`
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma Q(s', a') - Q(s, a) \right)
`$

Where:
- $`s`$ and $`a`$ are the current state and action.
- $`r`$ is the reward received after taking action $`a`$.
- $`s'`$ and $`a'`$ are the next state and the action chosen in that state.
- $`\alpha`$ is the learning rate, controlling how much new information overrides old estimates.
- $`\gamma`$ is the discount factor, balancing immediate and future rewards.

### Epsilon-Greedy Policy
The algorithm employs an epsilon-greedy policy for action selection:
- With probability $`\epsilon`$, the agent selects a random action to explore the environment.
- With probability $`1 - \epsilon`$, the agent selects the action with the highest Q-value to exploit its current knowledge.

---

## Results Explanation

### Cumulative Rewards Plot
The following plot shows the cumulative rewards achieved by the agent over the training episodes. The rewards increase as the agent learns to optimize its path and maximize the remaining profits at each target.

![Cumulative Reward Plot](https://github.com/MOONLABIISERB/marl-ecs-course/blob/Ankur_21043/Midsem/cumulative_reward_plot.png)

- **Moving Average**: To smooth out fluctuations in rewards across episodes, a moving average is calculated (window size = 100 episodes).

### Optimal Policy Plot
After training, the SARSA algorithm identifies the optimal path for visiting all targets. The plot below visualizes the agent's final learned policy, showing the sequence of target visits with arrows.

![Optimal Policy Plot](https://github.com/MOONLABIISERB/marl-ecs-course/blob/Ankur_21043/Midsem/optimal_path.png)

### Best Reward and Path
After training, the SARSA algorithm achieves the following best results:
- **Best Cumulative Reward**: `159.57`
- **Best Path**: `[0, 1, 2, 3, 5, 6, 8, 9, 7, 4]`

---

## Conclusion

The SARSA algorithm effectively solves the Traveling Salesman Problem in this custom environment, learning to balance exploration and exploitation to maximize profits by minimizing travel distances. The performance is demonstrated by the increasing cumulative rewards and the learned optimal path.
