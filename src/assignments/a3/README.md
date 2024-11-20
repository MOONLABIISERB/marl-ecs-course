# Assignment-03

## Deep Q-Learning on GridWorld Environment

This repository implements a **Deep Q-Learning (DQN)** algorithm to train agents in a custom **GridWorld** environment. The goal is to use reinforcement learning to train agents to navigate through a grid, avoiding obstacles and reaching their designated goal positions.

---

## Contents

1. [Overview](#overview)
2. [Environment Description](#environment-description)
3. [Algorithm Description](#algorithm-description)
4. [Code Structure](#code-structure)
5. [Usage](#usage)
6. [Results](#results)
7. [Mathematical Background](#mathematical-background)
8. [References](#references)

---

## Overview

The project simulates a grid-based environment where multiple agents aim to reach their respective goal positions. The environment includes:

- Obstacles (walls),
- A finite field of view for agents,
- Penalties for invalid moves and steps.

The Deep Q-Learning algorithm uses a neural network to approximate the Q-value function for decision-making. Agents learn optimal policies through exploration (random actions) and exploitation (choosing the best-known action).

---

## Environment Description

The **GridWorld Environment** is a $10 \times 10$ grid with the following features:

- **Agents**: 4 agents, each starting at a unique position.
- **Goals**: 4 goals, one for each agent.
- **Obstacles**: Fixed walls prevent certain movements.
- **Field of View**: Agents can only "see" a limited area ($2 \times 2$) around them.
- **Actions**: Agents can perform 5 actions: `NOPE` (stay), `UP`, `DOWN`, `LEFT`, and `RIGHT`.

| Feature          | Description                             |
| ---------------- | --------------------------------------- |
| Grid Size        | $10 \times 10$                          |
| Number of Agents | 4                                       |
| Agent FOV        | $2 \times 2$                            |
| Obstacles        | Fixed walls                             |
| Goals            | Each agent has one unique goal position |
| Max Steps        | 100                                     |

**Reward Mechanism**:

- **Reaching the goal**: $+50$ reward.
- **Invalid move**: $-10$ penalty.
- **Valid move**: $-1$ step penalty.
- **All agents reaching their goals**: Additional $+500$.

---

## Algorithm Description

Deep Q-Learning (DQN) is an enhancement of Q-Learning, using a neural network to estimate the action-value function $Q(s, a)$. The goal is to train the network to predict future rewards based on the current state and action.

**Steps in Training**:

1. **Initialization**:
   - Define the neural network $Q_\theta(s, a)$ and target network $Q'_\theta(s, a)$.
   - Initialize environment and parameters ($\epsilon, \gamma, \alpha$).
2. **Action Selection**:
   - Use $\epsilon$-greedy policy to balance exploration and exploitation.
3. **State Transition**:
   - Perform the action, observe the reward $r$ and next state $s'$.
4. **Update Rule**:
   $$
   y = r + \gamma \max_a Q'(s', a)
   $$
   $$
   \mathcal{L} = \frac{1}{2} \big(y - Q(s, a)\big)^2
   $$
   - Backpropagate the loss $\mathcal{L}$ and update the network.
5. **Target Network Update**:
   - Periodically update $Q'_\theta(s, a)$ with the weights of $Q_\theta(s, a)$.

---

## Code Structure

- **`env.py`**: Contains the implementation of the `GridWorldEnv` class, which simulates the environment.
- **`dqn.py`**: Defines the `Q_Network` class and implements the training loop.
- **`train.py`**: Integrates the environment and DQN to train agents.

---

### Running the Training Script

To train the agents using DQN:

```bash
python train.py
```

---

## Results

### Performance Metrics

We evaluated the performance of the agents based on the percentage of successful goal reaches and average rewards over multiple episodes.

| **Metric**             | **Value**     |
| ---------------------- | ------------- |
| Episodes Trained       | $500$         |
| Success Rate           | $78.9238\ \%$ |
| Average Reward/Episode | $-32.2456$    |

---

## Mathematical Background

### Q-Learning Update Rule

The Q-value function is updated iteratively as:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\big]
$$

In DQN, a neural network approximates $Q(s, a)$. The loss function minimizes the difference between predicted Q-values and target Q-values:

$$
\mathcal{L}(\theta) = \mathbb{E} \big[(y - Q_\theta(s, a))^2\big]
$$

Where:

- $y = r + \gamma \max_a Q'_\theta(s', a)$
