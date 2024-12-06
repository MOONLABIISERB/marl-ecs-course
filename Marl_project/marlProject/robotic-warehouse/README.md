## how to execute: run testlarge.py for large layout and testsmall.py for small layout

#### Project report has name "MARLProject_21008"

#### Objective
The goal of this project was for two agents to collaborate in moving requested shelves containing goods to
 goal positions and then returning the shelves to empty spots. The agents were required to minimize time
 and collisions in a 2D grid world while avoiding obstacles, which included non-requested shelves and the
 other agent.

 #### Method: DQN

 #### Program Setup:
 • main.py: Calls dql.py for training the agents.
 
 • dql.py: Implements and trains the DQN network.
 
 • testlarge.py: Tests the saved DQN models on the environment.

 #### Parameters Used:
 • Action space: 0, 1, 2, 3, 4
 
 • Observation space: 71 in length, providing information about surrounding grids.
 
 • Number of agents (nagents): 2
 
 • Reward type: Reward shaping was done with the following structure:
 
 – −0.4 for any action that is not an intermediate state.
 
 – −0.1 for picking up a requested shelf.
 
 – 2.0 for delivering a requested shelf to the goal.
 
 – 3.0 for delivering the shelf back to the empty spo
