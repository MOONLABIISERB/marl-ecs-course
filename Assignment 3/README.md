# Multi-Agent Path Finding (MAPF) Solver

This repository contains a Python implementation of a Multi-Agent Path Finding (MAPF) solver using a multi-agent reinforcement learning approach.

## Project Overview

The MAPF problem involves navigating multiple agents within a grid world environment, where each agent has a unique start position and a fixed goal location. The agents must reach their respective goals while avoiding obstacles and collisions with other agents.

This project implements a solution using a combination of Q-learning and multi-agent rollouts to find optimal paths for all agents, minimizing the overall time required for all agents to reach their destinations.

## Key Features

- Grid world environment with customizable size, obstacles, and agent start/goal positions
- Multi-agent Q-learning algorithm for learning coordinated movement policies
- Collision avoidance during training and execution
- Reward shaping to encourage reaching goals, minimizing path length, and avoiding collisions
- Visualization of the environment and the learned solution

## Training and Test:

The training process involves the following steps:

Initialize the GridWorld environment with the specified obstacles, agent start/goal positions, and action space.
Create an instance of the MAPFLearning class, which encapsulates the multi-agent Q-learning algorithm.
Train the agents using the train() method, which runs multiple episodes of learning and updates the Q-tables for each agent.
During training, the exploration rate (epsilon) is gradually decreased to transition from exploration to exploitation.
In Question 1:
We train the model in each question for 500000 episodes. Each episode runs for a maximum of 700 steps.

While, in Question 2 (Bonus):
We train the model in the bonus question for 500000 episodes with randomized start positions. Each episode runs for a maximum of 700 steps.

The training progress is periodically visualized to monitor the agents' performance.

After training, the learned policy can be executed using the execute_learned_policy() method. This will showcase the final solution, where the agents navigate to their goals while avoiding collisions.

## Results:

**We can observe from the data in Q1, the steps required to reach the solution is reached to as low Steps: 39 at Episode 432000.**

**We can observe from the data in Q2, the steps required to reach the solution is reached to as low Steps: 30 at Episode 399000.**

# Key Observations

## 1. Initial Behavior:
- **Both graphs**:
  - Start with high step counts around **700 steps per episode**
  - Show a sharp initial decline in the first ~50,000 episodes, indicating **rapid early learning**

## 2. Overall Trend:
- **Both graphs**:
  - Demonstrate a general downward trend, indicating **improving performance** (fewer steps needed per episode)
  - The decrease is steeper in the early stages and gradually levels off

## 3. Differences in Patterns:
### Graph 1 (Fixed Start):![Screenshot from 2024-11-17 05-40-44](https://github.com/user-attachments/assets/76cda286-cbbf-4fe4-89b5-48500ec315b3)

- Shows more **volatile behavior** with larger fluctuations
- Has more **pronounced spikes**, especially in the **100,000-200,000 episode** range
- Takes longer to stabilize despite fixed starting position
- Shows **greater variance** in performance
- **Best performance**: **39 steps** at Episode **432,000**
- Key characteristics:
  - More **dramatic fluctuations**
  - Agent might be **over-optimizing** for specific paths
  - More susceptibility to getting stuck in **local optima**
  - Potentially more **aggressive exploration strategy**
  - Fixed position might lead to more **specialized but unstable strategies**

### Graph 2 (Randomized Start): ![Screenshot from 2024-11-17 10-22-39](https://github.com/user-attachments/assets/c6667560-e0af-4f2e-a09a-6094ee9fe2fe)

- **Smoother overall decline** despite randomized starts, which is interesting
- More **consistent variance** throughout training
- Stabilizes around **250-300 steps** after Episode **300,000**
- Shows **remarkable stability** despite random starting positions
- **Best performance**: **30 steps** at Episode **399,000**

## 4. Learning Stability:
- **Graph 1** (Fixed Start):
  - Exhibits more **exploration behavior**, as evidenced by the larger fluctuations
- **Graph 2** (Randomized Start):
  - Shows more **stable learning** with smoother transitions
  - Suggests the agent developed a **more robust and generalized strategy** early on
  - Randomization might have **prevented the agent from getting stuck in local optima**

---

## Conclusion
1. **Randomization** acted as a form of **regularization**, leading to more **stable learning**
2. **Fixed starting position** allowed for more **specialized optimization**, but at the cost of stability
3. The **fixed position** (Graph 1) achieved **worse minimum performance** and with less consistency
4. The **randomized approach** (Graph 2) might actually be more reliable in practice, as reflected by:
   - **Better minimum steps** achieved
   - **Fewer episodes** required to achieve this
