# Multi-Agent Reinforcement Learning - Assignment 2
## Name: Aditya Mishra
## Roll Number: 21013

## Table of Contents
- [Overview](#overview)
- [Question 1: Travelling Salesman Problem (TSP)](#question-1-travelling-salesman-problem-tsp)
  - [Dynamic Programming (Value Iteration)](#dynamic-programming-value-iteration)
  - [Monte Carlo (First Visit)](#monte-carlo-first-visit)
  - [Issues Faced](#issues-faced)
- [Question 2: Sokoban Grid-World](#question-2-sokoban-grid-world)
  - [Dynamic Programming (Value Iteration)](#dynamic-programming-value-iteration-1)
  - [Monte Carlo (First Visit)](#monte-carlo-first-visit-1)
- [Results and Findings](#results-and-findings)
- [How to Run the Code](#how-to-run-the-code)
- [Conclusion](#conclusion)

## Overview
This assignment involved solving the Travelling Salesman Problem (TSP) and Sokoban grid-world puzzle using Dynamic Programming and Monte Carlo methods. Below are detailed discussions, findings, and instructions on how to run the code.

## Question 1: Travelling Salesman Problem (TSP)
The Travelling Salesman Problem (TSP) was modeled as a reinforcement learning problem, where the agent must visit a set number of targets while minimizing the total distance traveled.

### Dynamic Programming (Value Iteration)
The Value Iteration algorithm was applied to solve the TSP. The goal was to find the optimal policy that minimizes the total cost (distance) of the trip.

**Output:**
Value Iteration Policy: [0 0 0 0 0 0]
Value Iteration Values: [-9999.99901261 -9999.99902249 -9999.99902249 -9999.99902249
 -9999.99902249 -9999.99902249]

In the output, we observed that all policy values were set to 0, indicating a failure in the value iteration process. Additionally, the value function displayed extremely negative values, implying that the agent might be revisiting the same locations multiple times and incurring a large penalty for doing so.

### Monte Carlo (First Visit)
Monte Carlo first-visit method was implemented to explore episodes and calculate the state-value function. However, an error was encountered during execution.

**Error:**

Traceback (most recent call last):
  File "/Users/Aditya/Desktop/question1.py", line 167, in <module>
    V_mc = monte_carlo_first_visit(env)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/Aditya/Desktop/question1.py", line 151, in monte_carlo_first_visit
    V[state] = np.mean(returns[state])
    ~~~~~~~~^
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

**Issues Faced:**
Value Iteration: The Value Iteration method resulted in a policy that always suggested the agent should stay at the same location (policy [0, 0, 0, 0, 0, 0]). The large penalties for revisiting locations caused the value function to become overly negative. The reward structure was updated to use a smaller penalty for revisiting locations, but the issue persisted.
Monte Carlo: The IndexError in the Monte Carlo method occurred because the state was being indexed incorrectly. The state should be an integer or tuple that can be indexed in a dictionary, not a floating-point value or invalid index.

## Question 2: Sokoban Grid-World

The Sokoban puzzle was modeled as a grid-world environment where the agent moves boxes to their designated storage locations. The agent's objective was to solve the puzzle using Dynamic Programming and Monte Carlo methods.

### Dynamic Programming (Value Iteration)
The value iteration algorithm was applied to the Sokoban puzzle. The state space consisted of the agent’s position and the positions of the boxes.

**Output:**
Value Iteration Policy:
State: (0, 0), Best Action: 1
State: (0, 1), Best Action: 1
State: (0, 2), Best Action: 2
State: (0, 3), Best Action: 2
...
State: (5, 6), Best Action: 0

### Monte Carlo (First Visit)
Monte Carlo first-visit method was applied to estimate the state values. The algorithm successfully computed the values for each state after running multiple episodes.

### Monte Carlo First Visit Values:
State: (0, 0), Value: -44.095978054507995
State: (0, 1), Value: -38.340149974188606
State: (0, 2), Value: -34.23537632821142
State: (0, 3), Value: -29.71982284739425
...
State: (5, 6), Value: -20.042413639647247

**Results and Findings**

Dynamic Programming: The value iteration policy showed that the agent tends to follow a logical path based on the grid’s structure. The policy for each state points to actions (up, down, left, right) that lead the agent toward the boxes or storage locations.
Monte Carlo: The Monte Carlo method produced a negative value for each state, indicating the cost associated with each state-action pair. The values are useful for learning the best possible actions to minimize the number of moves.

## How to Run the Code

Requirements:
Python 3.x
numpy library
gymnasium for the RL environment

## Conclusion

In this assignment, we explored solving TSP and Sokoban using reinforcement learning techniques such as Dynamic Programming (Value Iteration) and Monte Carlo methods. While we encountered some issues in the TSP implementation (e.g., overly negative values and indexing errors), the Sokoban puzzle yielded more reasonable results in both approaches.

Further refinement of the reward structure and state indexing could enhance the performance of the TSP solution. The Sokoban results demonstrated the potential of reinforcement learning in solving grid-based puzzles.

This README provides an overview of the assignment, steps to replicate the code, and key findings. For further details, refer to the respective Python files.


