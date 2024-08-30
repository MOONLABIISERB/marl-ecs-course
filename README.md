# Multi-Agent Reinforcement Learning (MARL) Assignment 1

### Anshuman Dangwal 21044  

This repository contains the code and report for the MARL Assignment 1, focusing on two Markov Decision Processes (MDPs): the Student MDP and the Robot MDP.

## 1. Student MDP Problem

### Problem Description
The Student MDP models a student's day involving transitions between three states:
- Canteen
- Hostel
- Academic Building

Actions include:
- Attend Classes
- Hungry

Transition probabilities and rewards vary based on the current state and action taken. The goal is to derive the optimal policy and value function for the student.

### Key Components
- **States (S):** {'canteen', 'hostel', 'academic building'}
- **Actions (A):** {'attend classes', 'hungry'}
- **Reward Function (R):** {'canteen': 1, 'hostel': -1, 'academic building': 3}
- **Transition Probabilities (P):** Probabilistic transitions between states based on actions.

### Results
Both value iteration and policy iteration yielded the same optimal policy:
- **Optimal Policy:** Always attend classes.
- **Optimal Value Function:** {'canteen': 17.96, 'hostel': 15.19, 'academic building': 20.98}

## 2. Robot MDP Problem

### Problem Description
The Robot MDP models a robot navigating a 9x9 grid environment with obstacles. The robot can move in four directions: up, down, left, and right. The objective is to reach the goal state while avoiding walls.

### Key Components
- **States (S):** All positions (i, j) in the grid, excluding walls.
- **Actions (A):** {'up', 'down', 'left', 'right'}
- **Reward Function (R):** 1 for reaching the goal, 0 otherwise.
- **Transition Probabilities (P):** Deterministic transitions based on actions.

### Results
Both value iteration and policy iteration produced the same optimal policy and value function for navigating the grid.

### For Graph and Table of Student MDP and Quiver plot of Robot MDP please refer to the report marl_ass1_report.pdf
