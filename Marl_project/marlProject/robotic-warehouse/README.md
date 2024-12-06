## how to execute: run testlarge.py for large layout and testsmall.py for small layout

#### Project report has name "MARLProject_21008"

#### Objective
The goal of this project was for two agents to collaborate in moving requested shelves containing goods to
 goal positions and then returning the shelves to empty spots. The agents were required to minimize time
 and collisions in a 2D grid world while avoiding obstacles, which included non-requested shelves and the
 other agent.

 #### Program Setup:
 • main.py: Calls dql.py for training the agents.
 
 • dql.py: Implements and trains the DQN network.
 
 • testlarge.py: Tests the saved DQN models on the environment.
