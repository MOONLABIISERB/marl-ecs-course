# Team-Based Car Racing Using Multi-Agent Reinforcement Learning

This repository contains the implementation of a semester project for the ECS427: Multi-Agent Reinforcement Learning course. The project models a cooperative-competitive racing environment where teams of agents race against each other, balancing individual performance and team strategies.

## Authors
- Akshat Singh
- Gandhi Ananya Amit
- Chinmay Mundane

---

## Introduction

In professional racing sports, manufacturers aim to maximize team success by optimizing the performance of all their vehicles. Inspired by this, our project introduces a simulation where two teams of agents compete on a racing track. The goal is to have at least one agent from a team finish first, reflecting real-world dynamics of cooperation within competition.

Key Features:
- Cooperative and competitive gameplay.
- Two teams, each with two agents, racing on a circular track.
- The team wins if any of its agents completes a lap first.

---

## Environment Description

### Track
- **Size**: 30x30 grid with a circular track of width 5 cells.
- **Agents**: 4 agents (2 per team) begin at the starting line and complete one clockwise lap.

### Agent Actions
- Move: Left, Right, Up, Down.
- Stay in place.

### Rules
1. **Out of Bounds**: Agents stay in their current position if they move outside the track.
2. **Collision**: Agents remain in place for two steps if a collision occurs.
3. **Episode End**: Ends when any agent completes a lap.

### Observations
Each agent receives:
1. **Environment Matrix**: Cell types:
   - `0`: Unwalkable cells.
   - `1`: Walkable cells.
   - `2`: Agent's position.
   - `3`: Teammate's position.
   - `4`: Enemy positions.
2. **Angles**: Relative to the center for the agent, its teammate, and enemies.

---

## Reward Structure

To ensure balanced learning, the following rewards/penalties were implemented:
- **Step Penalty**: Discourages unnecessary movements.
- **Angle Reward**: Encourages clockwise progress.
- **Teammate Completion Reward**: Promotes cooperation within teams.
- **Angle Comparison Reward/Penalty**: Maintains competitive advantage.
- **Enemy Completion Penalty**: Penalty when an opponent finishes the lap.

---

## Algorithm

### Model Architecture
- **Input**: 904 features.
- **Hidden Layers**: 2 layers with 128 nodes each.
- **Output**: 5 possible actions.

### Learning Strategy
- **Algorithm**: Multi-Agent Deep Q-Network (MADQN).
- **Centralized Training, Centralized Execution (CTCE)**: Both Q-Network and Target Network operate on individual agent states and actions.

### Training Details
- Two agents train with epsilon = 0.1 and gamma = 0.9 using MADQN.
- The other two agents use a greedy approach (epsilon = 0, gamma = 0).

---

## Results

1. **Without Competition**:
   - Rewards increased but remained negative.

2. **With Random Policy Opponents**:
   - Agents struggled to distinguish clockwise vs. counterclockwise movement.

3. **With Greedy MADQN Opponents**:
   - Agents achieved stable learning with rewards converging closer to zero.

4. **Final Evaluation**:
   - After 1000 episodes, agents showed potential, with one agent consistently improving.

-------------

