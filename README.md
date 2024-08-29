# Multi-Agent Reinforcement Learning - Assignment 1

## Overview
This repository contains the report and related files for the first assignment in the Multi-Agent Reinforcement Learning course (ECS/DSE-427/627). The assignment involves designing a Markov Decision Process (MDP) for a student's activities on campus and solving a 9x9 grid-world environment using Value Iteration and Policy Iteration techniques.

## Contents
- **Report.pdf**: Detailed report explaining the methodology, implementation, and results of the assignment.
- **Code**: [Link to code, if any is provided]
- **Results**: Summarized results for the MDP and Grid-World tasks.

## Assignment Details

### Question 1: Designing and Solving a Finite MDP
- **Task**: Design a finite MDP to model the decision-making process of a student navigating between the hostel, academic building, and canteen.
- **Methods Used**: 
  - Value Iteration
  - Policy Iteration
- **Results**:
  - Optimal Value Functions and Policies were derived using both methods.
  - Differences in the value functions indicate how each method approaches optimization.

### Question 2: Solving the 9x9 Grid-World Environment
- **Task**: Solve a 9x9 grid-world problem where a robot must navigate from a starting position to a goal, considering tunnels acting as one-way portals.
- **Methods Used**:
  - Value Iteration
  - Policy Iteration
- **Results**:
  - Both methods produced identical value functions and optimal policies, guiding the agent effectively towards the goal.

## Results Summary
### Question 1:
- **Optimal Value Function (Value Iteration)**:
  - Hostel: 18.95
  - Academic Building: 20.94
  - Canteen: 19.81
- **Optimal Value Function (Policy Iteration)**:
  - Hostel: 13.10
  - Academic Building: 13.78
  - Canteen: 10.00
- **Optimal Policy**: Attending classes at all locations is the best course of action.

### Question 2:
- **Optimal Value Function (Value Iteration and Policy Iteration)**:
  - Identical value functions across the grid-world, with values increasing as the agent approaches the goal.
- **Optimal Policy**: Identical policies directing the agent to move towards the goal using the shortest path, taking into account the one-way tunnels.

## How to Use
- **Report**: Review the detailed explanations and results in the `Report.pdf` file.
- **Code**: Run the provided code (if applicable) to replicate the results or modify it for further exploration.

