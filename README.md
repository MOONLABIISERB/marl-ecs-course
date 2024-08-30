# marl-ecs-course

# RL_Assignment1

This Repository consist of solution to the assignment 1 for Multi-agent Reinforcement learning course.
In both the question we were supposed to solve and MDP by policy iteration and value iteration.

Simply run the code in jupyter notebook to recreate the results.

# Question 1

In this question the given MDP has been solved bu both value iteration and policy iteration, the optimal policy is to take the action
"attend" in all states.
The diagram for visualizing the MDP has been provided in the report.

# Question 2

In this question a maze was given and we were supposed to find an optimal path through the maze, this question can be modeled as an MDP and can thus be solved by methods like policy iteration and value iteration.
The problem can be formulated as a Markov Decision Process (MDP) as follows:

-> Each cell of the maze represents a distinct state in the MDP.<br>
-> State transitions are deterministic, meaning that each action leads to a specific next state with a probability of 1.<br>
-> The maze contains walls; any action that would lead to a state occupied by a wall results in the agent remaining in its current state.<br>
->There is a teleportation cell in the maze; any action taken in this cell instantly transports the agent to a different, predetermined cell. This teleportation works only one way.<br>
-> There is a reward of +1 on the final state and 0 for all other state.
