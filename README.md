# Assignment 1 and Assignment 2 Results
## Output and Results have been sent as jpg file in respective folders
# Assignment 1 (Folder name Q1 and Q2)
## Q1 Result:
#### Optimum Values
value(Hostel)= 16.05623156 

value(academic building)= 21.84650722 

value(canteen)= 18.82669839

#### Optimum policy

π(Hostel)= Study

π(academic building)= Study

π(canteen)= Study

## Q2 Result:

Value Iteration and Policy iteration values have been uploaded along with code and quiver plot

# Assignment2 (Folder name: Assignment2)
## Q1 a)
### Value Iteration result
```bash
Value iteration converged after 3 iterations
Optimal Path: [0, 2, 4, 1, 3]
```
#### Value iteration converts and output shows optimum policy along with optimum values.

## Q1 b)
### Monte Carlo result
#### Monte carlo using exploring starts was performed.


#### Value iteration converts and output shows optimum policy along with optimum values.

## Q2
#### a) Value Iteration Result: The grid is the same structure as the one provided in the assignment pdf. Value iteration result shows that the terminal state is reached after 10 iteration. The state is taken as a tuple of the human and the box state. The optimum policy and optimum state values of the state has been given in the output. 
### Traversal of states using optimum policy for initial state Human: (1, 2) and Box(4, 3) where end goal is (3,1) for box
```bash
Initial State: Human (1, 2), Box (4, 3)
Step 1: Human moves DOWN, New Human Pos: (2, 2), New Box Pos: (4, 3)
Step 2: Human moves DOWN, New Human Pos: (3, 2), New Box Pos: (4, 3)
Step 3: Human moves RIGHT, New Human Pos: (3, 3), New Box Pos: (4, 3)
Step 4: Human moves RIGHT, New Human Pos: (3, 4), New Box Pos: (4, 3)
Step 5: Human moves DOWN, New Human Pos: (4, 4), New Box Pos: (4, 3)
Step 6: Human moves LEFT, New Human Pos: (4, 3), New Box Pos: (4, 2)
Step 7: Human moves LEFT, New Human Pos: (4, 2), New Box Pos: (4, 1)
Step 8: Human moves DOWN, New Human Pos: (5, 2), New Box Pos: (4, 1)
Step 9: Human moves LEFT, New Human Pos: (5, 1), New Box Pos: (4, 1)
Step 10: Human moves UP, New Human Pos: (4, 1), New Box Pos: (3, 1)
Terminal State reached in 10 steps: Human (4, 1), Box (3, 1)
```

#### b) Monte Carlo result: After performing 100000 episodes Monte Carlo fails to find optimum policy, and the block state loops at the terminal state edge (4,4).
```bash
Initial State: Human (1, 2), Box (4, 3)
Step 1: Human moves DOWN, New Human Pos: (2, 2), New Box Pos: (4, 3)
Step 2: Human moves DOWN, New Human Pos: (3, 2), New Box Pos: (4, 3)
Step 3: Human moves DOWN, New Human Pos: (4, 2), New Box Pos: (4, 3)
Step 4: Human moves RIGHT, New Human Pos: (4, 3), New Box Pos: (4, 4)
No valid action found. Ending simulation.
```
### Difference between Value Iteration and Monte Carlo:
#### Value iteration converges quicker than Monte Carlo since the latter has an exploring nature, i.e exploring starts.



