# Assignment2 (Folder name: Assignment2)
## Q1 a)
### Value Iteration result and Optimum Policy
```bash
Value iteration converged after 3 iterations
Optimal Path: [0, 2, 4, 1, 3]

Optimum State Values and Policy after Convergence:
State (loc=0, visited_set=0b1): Value=-89.6165, Next City=2
State (loc=0, visited_set=0b11): Value=-68.4773, Next City=2
State (loc=0, visited_set=0b101): Value=-66.2466, Next City=1
State (loc=0, visited_set=0b111): Value=-30.9476, Next City=3
State (loc=0, visited_set=0b1001): Value=-76.3999, Next City=2
State (loc=0, visited_set=0b1011): Value=-54.0940, Next City=2
State (loc=0, visited_set=0b1101): Value=-51.8633, Next City=1
State (loc=0, visited_set=0b1111): Value=-15.1475, Next City=4
State (loc=0, visited_set=0b10001): Value=-57.8827, Next City=2
State (loc=0, visited_set=0b10011): Value=-44.6662, Next City=2
State (loc=0, visited_set=0b10101): Value=-42.7740, Next City=1
State (loc=0, visited_set=0b10111): Value=-16.5644, Next City=3
State (loc=0, visited_set=0b11001): Value=-31.7995, Next City=2
State (loc=0, visited_set=0b11011): Value=-30.5513, Next City=2
State (loc=0, visited_set=0b11101): Value=-29.5575, Next City=1
State (loc=0, visited_set=0b11111): Value=0.0000, Next City=0
State (loc=1, visited_set=0b10): Value=-98.0348, Next City=0
State (loc=1, visited_set=0b11): Value=-59.9635, Next City=4
State (loc=1, visited_set=0b110): Value=-60.5051, Next City=0
State (loc=1, visited_set=0b111): Value=-36.6891, Next City=4
State (loc=1, visited_set=0b1010): Value=-83.6515, Next City=0
State (loc=1, visited_set=0b1011): Value=-45.8486, Next City=4
State (loc=1, visited_set=0b1110): Value=-44.7049, Next City=0
State (loc=1, visited_set=0b1111): Value=-22.3059, Next City=4
State (loc=1, visited_set=0b10010): Value=-74.2236, Next City=0
State (loc=1, visited_set=0b10011): Value=-27.3314, Next City=3
State (loc=1, visited_set=0b10110): Value=-46.1218, Next City=0
State (loc=1, visited_set=0b10111): Value=-13.2165, Next City=3
State (loc=1, visited_set=0b11010): Value=-60.1088, Next City=0
State (loc=1, visited_set=0b11011): Value=-1.2482, Next City=2
State (loc=1, visited_set=0b11110): Value=-29.5575, Next City=0
State (loc=1, visited_set=0b11111): Value=0.0000, Next City=0
State (loc=2, visited_set=0b100): Value=-96.7979, Next City=0
State (loc=2, visited_set=0b101): Value=-59.0652, Next City=4
State (loc=2, visited_set=0b110): Value=-61.4990, Next City=0
State (loc=2, visited_set=0b111): Value=-37.9260, Next City=4
State (loc=2, visited_set=0b1100): Value=-82.4147, Next City=0
State (loc=2, visited_set=0b1101): Value=-45.8486, Next City=4
State (loc=2, visited_set=0b1110): Value=-45.6988, Next City=0
State (loc=2, visited_set=0b1111): Value=-23.5427, Next City=4
State (loc=2, visited_set=0b10100): Value=-73.3253, Next City=0
State (loc=2, visited_set=0b10101): Value=-27.3314, Next City=3
State (loc=2, visited_set=0b10110): Value=-47.1157, Next City=0
State (loc=2, visited_set=0b10111): Value=-14.1148, Next City=3
State (loc=2, visited_set=0b11100): Value=-60.1088, Next City=0
State (loc=2, visited_set=0b11101): Value=-1.2482, Next City=1
State (loc=2, visited_set=0b11110): Value=-30.5513, Next City=0
State (loc=2, visited_set=0b11111): Value=0.0000, Next City=0
State (loc=3, visited_set=0b1000): Value=-98.0348, Next City=4
State (loc=3, visited_set=0b1001): Value=-59.9635, Next City=2
State (loc=3, visited_set=0b1010): Value=-70.6584, Next City=0
State (loc=3, visited_set=0b1011): Value=-37.9260, Next City=4
State (loc=3, visited_set=0b1100): Value=-68.4277, Next City=0
State (loc=3, visited_set=0b1101): Value=-36.6891, Next City=4
State (loc=3, visited_set=0b1110): Value=-31.7119, Next City=0
State (loc=3, visited_set=0b1111): Value=-14.3832, Next City=4
State (loc=3, visited_set=0b11000): Value=-74.2236, Next City=2
State (loc=3, visited_set=0b11001): Value=-15.3630, Next City=2
State (loc=3, visited_set=0b11010): Value=-47.1157, Next City=0
State (loc=3, visited_set=0b11011): Value=-14.1148, Next City=2
State (loc=3, visited_set=0b11100): Value=-46.1218, Next City=0
State (loc=3, visited_set=0b11101): Value=-13.2165, Next City=1
State (loc=3, visited_set=0b11110): Value=-16.5644, Next City=0
State (loc=3, visited_set=0b11111): Value=0.0000, Next City=0
State (loc=4, visited_set=0b10000): Value=-96.8680, Next City=2
State (loc=4, visited_set=0b10001): Value=-50.8741, Next City=2
State (loc=4, visited_set=0b10010): Value=-70.6584, Next City=2
State (loc=4, visited_set=0b10011): Value=-37.6576, Next City=2
State (loc=4, visited_set=0b10100): Value=-68.4277, Next City=1
State (loc=4, visited_set=0b10101): Value=-35.5224, Next City=1
State (loc=4, visited_set=0b10110): Value=-31.7119, Next City=0
State (loc=4, visited_set=0b10111): Value=-14.3832, Next City=3
State (loc=4, visited_set=0b11000): Value=-83.6515, Next City=2
State (loc=4, visited_set=0b11001): Value=-24.7909, Next City=2
State (loc=4, visited_set=0b11010): Value=-54.0940, Next City=2
State (loc=4, visited_set=0b11011): Value=-23.5427, Next City=2
State (loc=4, visited_set=0b11100): Value=-51.8633, Next City=1
State (loc=4, visited_set=0b11101): Value=-22.3059, Next City=1
State (loc=4, visited_set=0b11110): Value=-15.1475, Next City=0
State (loc=4, visited_set=0b11111): Value=0.0000, Next City=0

```
#### Value iteration converts and output shows optimum policy along with optimum values.

## Q1 b)
### Monte Carlo result
#### Monte carlo using exploring starts was performed.

```bash
Optimal Path: [0, 0, 0, 0, 0, 0]

Optimum State Values and Policy after Exploration:
State (loc=0, visited_set=0b1): Value=0.0000, Next City=0
State (loc=0, visited_set=0b11): Value=0.0000, Next City=0
State (loc=0, visited_set=0b101): Value=0.0000, Next City=0
State (loc=0, visited_set=0b111): Value=0.0000, Next City=0
State (loc=0, visited_set=0b1001): Value=0.0000, Next City=0
State (loc=0, visited_set=0b1011): Value=0.0000, Next City=0
State (loc=0, visited_set=0b1101): Value=0.0000, Next City=0
State (loc=0, visited_set=0b1111): Value=0.0000, Next City=0
State (loc=0, visited_set=0b10001): Value=0.0000, Next City=0
State (loc=0, visited_set=0b10011): Value=0.0000, Next City=0
State (loc=0, visited_set=0b10101): Value=0.0000, Next City=0
State (loc=0, visited_set=0b10111): Value=0.0000, Next City=0
State (loc=0, visited_set=0b11001): Value=0.0000, Next City=0
State (loc=0, visited_set=0b11011): Value=0.0000, Next City=0
State (loc=0, visited_set=0b11101): Value=0.0000, Next City=0
State (loc=0, visited_set=0b11111): Value=0.0000, Next City=0
State (loc=0, visited_set=0b100001): Value=0.0000, Next City=0
State (loc=0, visited_set=0b100011): Value=0.0000, Next City=0
State (loc=0, visited_set=0b100101): Value=0.0000, Next City=0
State (loc=0, visited_set=0b100111): Value=0.0000, Next City=0
State (loc=0, visited_set=0b101001): Value=0.0000, Next City=0
State (loc=0, visited_set=0b101011): Value=0.0000, Next City=0
State (loc=0, visited_set=0b101101): Value=0.0000, Next City=0
State (loc=0, visited_set=0b101111): Value=0.0000, Next City=0
State (loc=0, visited_set=0b110001): Value=0.0000, Next City=0
State (loc=0, visited_set=0b110011): Value=0.0000, Next City=0
State (loc=0, visited_set=0b110101): Value=0.0000, Next City=0
State (loc=0, visited_set=0b110111): Value=0.0000, Next City=0
State (loc=0, visited_set=0b111001): Value=0.0000, Next City=0
State (loc=0, visited_set=0b111011): Value=0.0000, Next City=0
State (loc=0, visited_set=0b111101): Value=0.0000, Next City=0
State (loc=0, visited_set=0b111111): Value=0.0000, Next City=0
State (loc=1, visited_set=0b10): Value=0.0000, Next City=0
State (loc=1, visited_set=0b11): Value=0.0000, Next City=0
State (loc=1, visited_set=0b110): Value=0.0000, Next City=0
State (loc=1, visited_set=0b111): Value=0.0000, Next City=0
State (loc=1, visited_set=0b1010): Value=0.0000, Next City=0
State (loc=1, visited_set=0b1011): Value=0.0000, Next City=0
State (loc=1, visited_set=0b1110): Value=0.0000, Next City=0
State (loc=1, visited_set=0b1111): Value=0.0000, Next City=0
State (loc=1, visited_set=0b10010): Value=0.0000, Next City=0
State (loc=1, visited_set=0b10011): Value=0.0000, Next City=0
State (loc=1, visited_set=0b10110): Value=0.0000, Next City=0
State (loc=1, visited_set=0b10111): Value=0.0000, Next City=0
State (loc=1, visited_set=0b11010): Value=0.0000, Next City=0
State (loc=1, visited_set=0b11011): Value=0.0000, Next City=0
State (loc=1, visited_set=0b11110): Value=0.0000, Next City=0
State (loc=1, visited_set=0b11111): Value=0.0000, Next City=0
State (loc=1, visited_set=0b100010): Value=0.0000, Next City=0
State (loc=1, visited_set=0b100011): Value=0.0000, Next City=0
State (loc=1, visited_set=0b100110): Value=0.0000, Next City=0
State (loc=1, visited_set=0b100111): Value=0.0000, Next City=0
State (loc=1, visited_set=0b101010): Value=0.0000, Next City=0
State (loc=1, visited_set=0b101011): Value=0.0000, Next City=0
State (loc=1, visited_set=0b101110): Value=0.0000, Next City=0
State (loc=1, visited_set=0b101111): Value=0.0000, Next City=0
State (loc=1, visited_set=0b110010): Value=0.0000, Next City=0
State (loc=1, visited_set=0b110011): Value=0.0000, Next City=0
State (loc=1, visited_set=0b110110): Value=0.0000, Next City=0
State (loc=1, visited_set=0b110111): Value=0.0000, Next City=0
State (loc=1, visited_set=0b111010): Value=0.0000, Next City=0
State (loc=1, visited_set=0b111011): Value=0.0000, Next City=0
State (loc=1, visited_set=0b111110): Value=0.0000, Next City=0
State (loc=1, visited_set=0b111111): Value=0.0000, Next City=0
State (loc=2, visited_set=0b100): Value=0.0000, Next City=0
State (loc=2, visited_set=0b101): Value=0.0000, Next City=0
State (loc=2, visited_set=0b110): Value=0.0000, Next City=0
State (loc=2, visited_set=0b111): Value=0.0000, Next City=0
State (loc=2, visited_set=0b1100): Value=0.0000, Next City=0
State (loc=2, visited_set=0b1101): Value=0.0000, Next City=0
State (loc=2, visited_set=0b1110): Value=0.0000, Next City=0
State (loc=2, visited_set=0b1111): Value=0.0000, Next City=0
State (loc=2, visited_set=0b10100): Value=0.0000, Next City=0
State (loc=2, visited_set=0b10101): Value=0.0000, Next City=0
State (loc=2, visited_set=0b10110): Value=0.0000, Next City=0
State (loc=2, visited_set=0b10111): Value=0.0000, Next City=0
State (loc=2, visited_set=0b11100): Value=0.0000, Next City=0
State (loc=2, visited_set=0b11101): Value=0.0000, Next City=0
State (loc=2, visited_set=0b11110): Value=0.0000, Next City=0
State (loc=2, visited_set=0b11111): Value=0.0000, Next City=0
State (loc=2, visited_set=0b100100): Value=0.0000, Next City=0
State (loc=2, visited_set=0b100101): Value=0.0000, Next City=0
State (loc=2, visited_set=0b100110): Value=0.0000, Next City=0
State (loc=2, visited_set=0b100111): Value=0.0000, Next City=0
State (loc=2, visited_set=0b101100): Value=0.0000, Next City=0
State (loc=2, visited_set=0b101101): Value=0.0000, Next City=0
State (loc=2, visited_set=0b101110): Value=0.0000, Next City=0
State (loc=2, visited_set=0b101111): Value=0.0000, Next City=0
State (loc=2, visited_set=0b110100): Value=0.0000, Next City=0
State (loc=2, visited_set=0b110101): Value=0.0000, Next City=0
State (loc=2, visited_set=0b110110): Value=0.0000, Next City=0
State (loc=2, visited_set=0b110111): Value=0.0000, Next City=0
State (loc=2, visited_set=0b111100): Value=0.0000, Next City=0
State (loc=2, visited_set=0b111101): Value=0.0000, Next City=0
State (loc=2, visited_set=0b111110): Value=0.0000, Next City=0
State (loc=2, visited_set=0b111111): Value=0.0000, Next City=0
State (loc=3, visited_set=0b1000): Value=0.0000, Next City=0
State (loc=3, visited_set=0b1001): Value=0.0000, Next City=0
State (loc=3, visited_set=0b1010): Value=0.0000, Next City=0
State (loc=3, visited_set=0b1011): Value=0.0000, Next City=0
State (loc=3, visited_set=0b1100): Value=0.0000, Next City=0
State (loc=3, visited_set=0b1101): Value=0.0000, Next City=0
State (loc=3, visited_set=0b1110): Value=0.0000, Next City=0
State (loc=3, visited_set=0b1111): Value=0.0000, Next City=0
State (loc=3, visited_set=0b11000): Value=0.0000, Next City=0
State (loc=3, visited_set=0b11001): Value=0.0000, Next City=0
State (loc=3, visited_set=0b11010): Value=0.0000, Next City=0
State (loc=3, visited_set=0b11011): Value=0.0000, Next City=0
State (loc=3, visited_set=0b11100): Value=0.0000, Next City=0
State (loc=3, visited_set=0b11101): Value=0.0000, Next City=0
State (loc=3, visited_set=0b11110): Value=0.0000, Next City=0
State (loc=3, visited_set=0b11111): Value=0.0000, Next City=0
State (loc=3, visited_set=0b101000): Value=0.0000, Next City=0
State (loc=3, visited_set=0b101001): Value=0.0000, Next City=0
State (loc=3, visited_set=0b101010): Value=0.0000, Next City=0
State (loc=3, visited_set=0b101011): Value=0.0000, Next City=0
State (loc=3, visited_set=0b101100): Value=0.0000, Next City=0
State (loc=3, visited_set=0b101101): Value=0.0000, Next City=0
State (loc=3, visited_set=0b101110): Value=0.0000, Next City=0
State (loc=3, visited_set=0b101111): Value=0.0000, Next City=0
State (loc=3, visited_set=0b111000): Value=0.0000, Next City=0
State (loc=3, visited_set=0b111001): Value=0.0000, Next City=0
State (loc=3, visited_set=0b111010): Value=0.0000, Next City=0
State (loc=3, visited_set=0b111011): Value=0.0000, Next City=0
State (loc=3, visited_set=0b111100): Value=0.0000, Next City=0
State (loc=3, visited_set=0b111101): Value=0.0000, Next City=0
State (loc=3, visited_set=0b111110): Value=0.0000, Next City=0
State (loc=3, visited_set=0b111111): Value=0.0000, Next City=0
State (loc=4, visited_set=0b10000): Value=0.0000, Next City=0
State (loc=4, visited_set=0b10001): Value=0.0000, Next City=0
State (loc=4, visited_set=0b10010): Value=0.0000, Next City=0
State (loc=4, visited_set=0b10011): Value=0.0000, Next City=0
State (loc=4, visited_set=0b10100): Value=0.0000, Next City=0
State (loc=4, visited_set=0b10101): Value=0.0000, Next City=0
State (loc=4, visited_set=0b10110): Value=0.0000, Next City=0
State (loc=4, visited_set=0b10111): Value=0.0000, Next City=0
State (loc=4, visited_set=0b11000): Value=0.0000, Next City=0
State (loc=4, visited_set=0b11001): Value=0.0000, Next City=0
State (loc=4, visited_set=0b11010): Value=0.0000, Next City=0
State (loc=4, visited_set=0b11011): Value=0.0000, Next City=0
State (loc=4, visited_set=0b11100): Value=0.0000, Next City=0
State (loc=4, visited_set=0b11101): Value=0.0000, Next City=0
State (loc=4, visited_set=0b11110): Value=0.0000, Next City=0
State (loc=4, visited_set=0b11111): Value=0.0000, Next City=0
State (loc=4, visited_set=0b110000): Value=0.0000, Next City=0
State (loc=4, visited_set=0b110001): Value=0.0000, Next City=0
State (loc=4, visited_set=0b110010): Value=0.0000, Next City=0
State (loc=4, visited_set=0b110011): Value=0.0000, Next City=0
State (loc=4, visited_set=0b110100): Value=0.0000, Next City=0
State (loc=4, visited_set=0b110101): Value=0.0000, Next City=0
State (loc=4, visited_set=0b110110): Value=0.0000, Next City=0
State (loc=4, visited_set=0b110111): Value=0.0000, Next City=0
State (loc=4, visited_set=0b111000): Value=0.0000, Next City=0
State (loc=4, visited_set=0b111001): Value=0.0000, Next City=0
State (loc=4, visited_set=0b111010): Value=0.0000, Next City=0
State (loc=4, visited_set=0b111011): Value=0.0000, Next City=0
State (loc=4, visited_set=0b111100): Value=0.0000, Next City=0
State (loc=4, visited_set=0b111101): Value=0.0000, Next City=0
State (loc=4, visited_set=0b111110): Value=0.0000, Next City=0
State (loc=4, visited_set=0b111111): Value=0.0000, Next City=0
State (loc=5, visited_set=0b100000): Value=0.0000, Next City=0
State (loc=5, visited_set=0b100001): Value=0.0000, Next City=0
State (loc=5, visited_set=0b100010): Value=0.0000, Next City=0
State (loc=5, visited_set=0b100011): Value=0.0000, Next City=0
State (loc=5, visited_set=0b100100): Value=0.0000, Next City=0
State (loc=5, visited_set=0b100101): Value=0.0000, Next City=0
State (loc=5, visited_set=0b100110): Value=0.0000, Next City=0
State (loc=5, visited_set=0b100111): Value=0.0000, Next City=0
State (loc=5, visited_set=0b101000): Value=0.0000, Next City=0
State (loc=5, visited_set=0b101001): Value=0.0000, Next City=0
State (loc=5, visited_set=0b101010): Value=0.0000, Next City=0
State (loc=5, visited_set=0b101011): Value=0.0000, Next City=0
State (loc=5, visited_set=0b101100): Value=0.0000, Next City=0
State (loc=5, visited_set=0b101101): Value=0.0000, Next City=0
State (loc=5, visited_set=0b101110): Value=0.0000, Next City=0
State (loc=5, visited_set=0b101111): Value=0.0000, Next City=0
State (loc=5, visited_set=0b110000): Value=0.0000, Next City=0
State (loc=5, visited_set=0b110001): Value=0.0000, Next City=0
State (loc=5, visited_set=0b110010): Value=0.0000, Next City=0
State (loc=5, visited_set=0b110011): Value=0.0000, Next City=0
State (loc=5, visited_set=0b110100): Value=0.0000, Next City=0
State (loc=5, visited_set=0b110101): Value=0.0000, Next City=0
State (loc=5, visited_set=0b110110): Value=0.0000, Next City=0
State (loc=5, visited_set=0b110111): Value=0.0000, Next City=0
State (loc=5, visited_set=0b111000): Value=0.0000, Next City=0
State (loc=5, visited_set=0b111001): Value=0.0000, Next City=0
State (loc=5, visited_set=0b111010): Value=0.0000, Next City=0
State (loc=5, visited_set=0b111011): Value=0.0000, Next City=0
State (loc=5, visited_set=0b111100): Value=0.0000, Next City=0
State (loc=5, visited_set=0b111101): Value=0.0000, Next City=0
State (loc=5, visited_set=0b111110): Value=0.0000, Next City=0
State (loc=5, visited_set=0b111111): Value=0.0000, Next City=0
```


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
## Optimal Policy:
```bash
State ((1, 1), (1, 2)): Take action UP
State ((1, 1), (2, 1)): Take action DOWN
State ((1, 1), (2, 2)): Take action RIGHT
State ((1, 1), (3, 2)): Take action DOWN
State ((1, 1), (3, 3)): Take action DOWN
State ((1, 1), (3, 4)): Take action UP
State ((1, 1), (4, 1)): Take action DOWN
State ((1, 1), (4, 2)): Take action DOWN
State ((1, 1), (4, 3)): Take action DOWN
State ((1, 1), (4, 4)): Take action UP
State ((1, 1), (5, 1)): Take action UP
State ((1, 1), (5, 2)): Take action UP
State ((1, 2), (1, 1)): Take action UP
State ((1, 2), (2, 1)): Take action LEFT
State ((1, 2), (2, 2)): Take action DOWN
State ((1, 2), (3, 2)): Take action DOWN
State ((1, 2), (3, 3)): Take action DOWN
State ((1, 2), (3, 4)): Take action UP
State ((1, 2), (4, 1)): Take action DOWN
State ((1, 2), (4, 2)): Take action DOWN
State ((1, 2), (4, 3)): Take action DOWN
State ((1, 2), (4, 4)): Take action UP
State ((1, 2), (5, 1)): Take action UP
State ((1, 2), (5, 2)): Take action UP
State ((2, 1), (1, 1)): Take action UP
State ((2, 1), (1, 2)): Take action UP
State ((2, 1), (2, 2)): Take action UP
State ((2, 1), (3, 2)): Take action DOWN
State ((2, 1), (3, 3)): Take action DOWN
State ((2, 1), (3, 4)): Take action UP
State ((2, 1), (4, 1)): Take action DOWN
State ((2, 1), (4, 2)): Take action DOWN
State ((2, 1), (4, 3)): Take action DOWN
State ((2, 1), (4, 4)): Take action UP
State ((2, 1), (5, 1)): Take action UP
State ((2, 1), (5, 2)): Take action UP
State ((2, 2), (1, 1)): Take action UP
State ((2, 2), (1, 2)): Take action UP
State ((2, 2), (2, 1)): Take action UP
State ((2, 2), (3, 2)): Take action DOWN
State ((2, 2), (3, 3)): Take action DOWN
State ((2, 2), (3, 4)): Take action UP
State ((2, 2), (4, 1)): Take action DOWN
State ((2, 2), (4, 2)): Take action DOWN
State ((2, 2), (4, 3)): Take action DOWN
State ((2, 2), (4, 4)): Take action UP
State ((2, 2), (5, 1)): Take action UP
State ((2, 2), (5, 2)): Take action UP
State ((3, 1), (1, 1)): Take action UP
State ((3, 1), (1, 2)): Take action UP
State ((3, 1), (2, 1)): Take action RIGHT
State ((3, 1), (2, 2)): Take action UP
State ((3, 1), (3, 2)): Take action DOWN
State ((3, 1), (3, 3)): Take action DOWN
State ((3, 1), (3, 4)): Take action UP
State ((3, 1), (4, 1)): Take action RIGHT
State ((3, 1), (4, 2)): Take action DOWN
State ((3, 1), (4, 3)): Take action RIGHT
State ((3, 1), (4, 4)): Take action UP
State ((3, 1), (5, 1)): Take action UP
State ((3, 1), (5, 2)): Take action UP
State ((3, 2), (1, 1)): Take action UP
State ((3, 2), (1, 2)): Take action UP
State ((3, 2), (2, 1)): Take action UP
State ((3, 2), (2, 2)): Take action LEFT
State ((3, 2), (3, 3)): Take action DOWN
State ((3, 2), (3, 4)): Take action UP
State ((3, 2), (4, 1)): Take action DOWN
State ((3, 2), (4, 2)): Take action RIGHT
State ((3, 2), (4, 3)): Take action RIGHT
State ((3, 2), (4, 4)): Take action UP
State ((3, 2), (5, 1)): Take action UP
State ((3, 2), (5, 2)): Take action UP
State ((3, 3), (1, 1)): Take action UP
State ((3, 3), (1, 2)): Take action UP
State ((3, 3), (2, 1)): Take action LEFT
State ((3, 3), (2, 2)): Take action LEFT
State ((3, 3), (3, 2)): Take action LEFT
State ((3, 3), (3, 4)): Take action UP
State ((3, 3), (4, 1)): Take action DOWN
State ((3, 3), (4, 2)): Take action DOWN
State ((3, 3), (4, 3)): Take action RIGHT
State ((3, 3), (4, 4)): Take action UP
State ((3, 3), (5, 1)): Take action UP
State ((3, 3), (5, 2)): Take action UP
State ((3, 4), (1, 1)): Take action UP
State ((3, 4), (1, 2)): Take action UP
State ((3, 4), (2, 1)): Take action LEFT
State ((3, 4), (2, 2)): Take action LEFT
State ((3, 4), (3, 2)): Take action LEFT
State ((3, 4), (3, 3)): Take action LEFT
State ((3, 4), (4, 1)): Take action DOWN
State ((3, 4), (4, 2)): Take action DOWN
State ((3, 4), (4, 3)): Take action DOWN
State ((3, 4), (4, 4)): Take action UP
State ((3, 4), (5, 1)): Take action UP
State ((3, 4), (5, 2)): Take action UP
State ((4, 1), (1, 1)): Take action UP
State ((4, 1), (1, 2)): Take action UP
State ((4, 1), (2, 1)): Take action UP
State ((4, 1), (2, 2)): Take action UP
State ((4, 1), (3, 2)): Take action RIGHT
State ((4, 1), (3, 3)): Take action RIGHT
State ((4, 1), (3, 4)): Take action UP
State ((4, 1), (4, 2)): Take action DOWN
State ((4, 1), (4, 3)): Take action UP
State ((4, 1), (4, 4)): Take action UP
State ((4, 1), (5, 1)): Take action UP
State ((4, 1), (5, 2)): Take action UP
State ((4, 2), (1, 1)): Take action UP
State ((4, 2), (1, 2)): Take action UP
State ((4, 2), (2, 1)): Take action UP
State ((4, 2), (2, 2)): Take action UP
State ((4, 2), (3, 2)): Take action RIGHT
State ((4, 2), (3, 3)): Take action RIGHT
State ((4, 2), (3, 4)): Take action UP
State ((4, 2), (4, 1)): Take action DOWN
State ((4, 2), (4, 3)): Take action UP
State ((4, 2), (4, 4)): Take action UP
State ((4, 2), (5, 1)): Take action UP
State ((4, 2), (5, 2)): Take action UP
State ((4, 3), (1, 1)): Take action UP
State ((4, 3), (1, 2)): Take action UP
State ((4, 3), (2, 1)): Take action UP
State ((4, 3), (2, 2)): Take action UP
State ((4, 3), (3, 2)): Take action UP
State ((4, 3), (3, 3)): Take action RIGHT
State ((4, 3), (3, 4)): Take action UP
State ((4, 3), (4, 1)): Take action LEFT
State ((4, 3), (4, 2)): Take action LEFT
State ((4, 3), (4, 4)): Take action UP
State ((4, 3), (5, 1)): Take action UP
State ((4, 3), (5, 2)): Take action UP
State ((4, 4), (1, 1)): Take action UP
State ((4, 4), (1, 2)): Take action UP
State ((4, 4), (2, 1)): Take action UP
State ((4, 4), (2, 2)): Take action UP
State ((4, 4), (3, 2)): Take action UP
State ((4, 4), (3, 3)): Take action UP
State ((4, 4), (3, 4)): Take action UP
State ((4, 4), (4, 1)): Take action LEFT
State ((4, 4), (4, 2)): Take action LEFT
State ((4, 4), (4, 3)): Take action LEFT
State ((4, 4), (5, 1)): Take action UP
State ((4, 4), (5, 2)): Take action UP
State ((5, 1), (1, 1)): Take action UP
State ((5, 1), (1, 2)): Take action UP
State ((5, 1), (2, 1)): Take action UP
State ((5, 1), (2, 2)): Take action UP
State ((5, 1), (3, 2)): Take action UP
State ((5, 1), (3, 3)): Take action UP
State ((5, 1), (3, 4)): Take action UP
State ((5, 1), (4, 1)): Take action UP
State ((5, 1), (4, 2)): Take action RIGHT
State ((5, 1), (4, 3)): Take action UP
State ((5, 1), (4, 4)): Take action UP
State ((5, 1), (5, 2)): Take action UP
State ((5, 2), (1, 1)): Take action UP
State ((5, 2), (1, 2)): Take action UP
State ((5, 2), (2, 1)): Take action UP
State ((5, 2), (2, 2)): Take action UP
State ((5, 2), (3, 2)): Take action UP
State ((5, 2), (3, 3)): Take action UP
State ((5, 2), (3, 4)): Take action UP
State ((5, 2), (4, 1)): Take action LEFT
State ((5, 2), (4, 2)): Take action UP
State ((5, 2), (4, 3)): Take action UP
State ((5, 2), (4, 4)): Take action UP
State ((5, 2), (5, 1)): Take action UP
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




