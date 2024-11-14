# assignment-1

## Summary

This assignment involves implementing Value Iteration and Policy Iteration for two scenarios:

1. **Campus MDP**: A finite MDP was created to model a student’s movements between three campus locations (Hostel, Academic Building, and Canteen), with actions and rewards assigned for each state and transition. Value Iteration and Policy Iteration algorithms were used to compute optimal state values and policies.

2. **9x9 Grid-World**: In a grid-world environment, a robot navigates from a starting position to a goal, using portals as shortcuts. We implemented Value Iteration and Policy Iteration to find the optimal navigation policy, visualized using quiver plots.

## Discussion

Both Value Iteration and Policy Iteration yielded similar optimal policies in each scenario, confirming the consistency of these methods in finding effective strategies:

- In the Campus MDP, both methods directed the student toward higher-reward locations.
- In the Grid-World, both methods found the shortest path to the goal, effectively utilizing portals.

The results validate the reliability of these methods in sparse reward settings, with Value Iteration converging faster but both methods ultimately providing comparable solutions.

## How to Run the Code

1. **Setup**:
   - Install required packages: `numpy` and `matplotlib`.
   ```bash
   pip install numpy matplotlib
   ```

2. **Running the Code**:
   - For the Campus MDP (Question 1) execute:
     ```bash
     campus_mdp.ipynb
     ```
   - For the 9x9 Grid-World (Question 2) execute:
     ```bash
     grid_world.ipynb
     ```

Each script will output optimal policies and, for the grid-world, a quiver plot visualizing the policies.
