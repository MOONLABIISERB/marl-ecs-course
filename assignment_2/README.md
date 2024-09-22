# ASSIGNMENT 2

## QUESTION-1
---------------

## QUESTION-2
Value Iteration:
theta = 1e-3
gamma = 0.9
Player reached target state after 10 steps.

Steps taken are as follows-

```bash
Testing Policy
Initial State: Player (1, 2), Box (4, 3)
Step 1: Player moved DOWN
Step 2: Player moved DOWN
Step 3: Player moved RIGHT
Step 4: Player moved RIGHT
Step 5: Player moved DOWN
Step 6: Player moved LEFT
Step 7: Player moved LEFT
Step 8: Player moved DOWN
Step 9: Player moved LEFT
Step 10: Player moved UP
Terminal State reached: Player (4, 1), Box (3, 1)
```


Monte Carlo:
Number of episodes for first visit and every visit methods = 1000
gamma = 0.9

The best action for each state are as-

```bash
Improved Policy from First-Visit Monte Carlo:
State: ((5, 1), ((4, 1),)), Best Action: UP
State: ((5, 2), ((4, 1),)), Best Action: LEFT
State: ((4, 2), ((4, 1),)), Best Action: DOWN
State: ((4, 3), ((4, 1),)), Best Action: LEFT
State: ((4, 4), ((4, 1),)), Best Action: LEFT
State: ((3, 3), ((4, 1),)), Best Action: DOWN
State: ((3, 2), ((4, 1),)), Best Action: DOWN
State: ((3, 1), ((4, 1),)), Best Action: DOWN
State: ((2, 2), ((4, 1),)), Best Action: DOWN
State: ((2, 1), ((4, 1),)), Best Action: RIGHT
State: ((4, 3), ((4, 2),)), Best Action: LEFT
State: ((3, 3), ((4, 2),)), Best Action: DOWN
State: ((4, 4), ((4, 2),)), Best Action: LEFT
State: ((4, 4), ((4, 3),)), Best Action: LEFT
State: ((3, 4), ((4, 3),)), Best Action: DOWN
State: ((3, 3), ((4, 3),)), Best Action: RIGHT
State: ((3, 2), ((4, 3),)), Best Action: RIGHT
State: ((3, 1), ((4, 3),)), Best Action: RIGHT
State: ((4, 1), ((4, 3),)), Best Action: UP
State: ((5, 1), ((4, 3),)), Best Action: RIGHT
State: ((5, 2), ((4, 3),)), Best Action: UP
State: ((4, 2), ((4, 3),)), Best Action: RIGHT
State: ((2, 2), ((4, 3),)), Best Action: DOWN
State: ((2, 1), ((4, 3),)), Best Action: RIGHT
State: ((1, 1), ((4, 3),)), Best Action: RIGHT
State: ((1, 2), ((4, 3),)), Best Action: DOWN
State: ((3, 2), ((4, 4),)), Best Action: DOWN
State: ((3, 3), ((4, 4),)), Best Action: LEFT
State: ((3, 4), ((4, 4),)), Best Action: LEFT
State: ((4, 2), ((4, 4),)), Best Action: UP
State: ((2, 2), ((4, 4),)), Best Action: DOWN
State: ((3, 1), ((4, 4),)), Best Action: RIGHT
State: ((1, 2), ((4, 4),)), Best Action: DOWN
State: ((2, 1), ((4, 4),)), Best Action: DOWN
State: ((4, 3), ((4, 4),)), Best Action: LEFT
State: ((5, 2), ((4, 4),)), Best Action: UP
State: ((4, 1), ((4, 4),)), Best Action: RIGHT
State: ((1, 1), ((4, 4),)), Best Action: DOWN
State: ((5, 1), ((4, 4),)), Best Action: UP
State: ((4, 3), ((5, 1),)), Best Action: LEFT
State: ((4, 2), ((5, 1),)), Best Action: UP
State: ((4, 1), ((5, 1),)), Best Action: RIGHT
State: ((5, 2), ((5, 1),)), Best Action: UP
State: ((3, 2), ((5, 1),)), Best Action: DOWN
State: ((3, 1), ((5, 1),)), Best Action: RIGHT
State: ((2, 1), ((5, 1),)), Best Action: LEFT
State: ((3, 3), ((5, 1),)), Best Action: LEFT
State: ((2, 2), ((5, 1),)), Best Action: DOWN
State: ((1, 2), ((5, 1),)), Best Action: DOWN
State: ((1, 1), ((5, 1),)), Best Action: DOWN
State: ((3, 4), ((5, 1),)), Best Action: LEFT
State: ((4, 4), ((5, 1),)), Best Action: LEFT
State: ((3, 4), ((4, 1),)), Best Action: DOWN
State: ((4, 2), ((5, 2),)), Best Action: UP
State: ((3, 2), ((5, 2),)), Best Action: LEFT
State: ((2, 2), ((5, 2),)), Best Action: DOWN
State: ((3, 3), ((5, 2),)), Best Action: LEFT
State: ((4, 3), ((5, 2),)), Best Action: UP
State: ((3, 4), ((5, 2),)), Best Action: LEFT
State: ((4, 4), ((5, 2),)), Best Action: LEFT
State: ((3, 1), ((5, 2),)), Best Action: RIGHT
State: ((2, 1), ((5, 2),)), Best Action: DOWN
State: ((1, 1), ((5, 2),)), Best Action: DOWN
State: ((1, 2), ((5, 2),)), Best Action: DOWN
State: ((4, 1), ((5, 2),)), Best Action: UP
State: ((5, 1), ((5, 2),)), Best Action: UP
State: ((3, 2), ((4, 2),)), Best Action: DOWN
State: ((3, 4), ((4, 2),)), Best Action: DOWN
State: ((1, 1), ((4, 1),)), Best Action: DOWN
State: ((1, 2), ((4, 1),)), Best Action: DOWN
State: ((4, 1), ((4, 2),)), Best Action: DOWN
State: ((3, 1), ((4, 2),)), Best Action: DOWN
State: ((2, 1), ((4, 2),)), Best Action: UP
State: ((1, 1), ((4, 2),)), Best Action: UP
State: ((2, 2), ((4, 2),)), Best Action: LEFT
State: ((1, 2), ((4, 2),)), Best Action: LEFT
State: ((5, 1), ((4, 2),)), Best Action: RIGHT
State: ((5, 2), ((4, 2),)), Best Action: UP
State: ((3, 2), ((3, 4),)), Best Action: UP
State: ((2, 2), ((3, 4),)), Best Action: DOWN
State: ((2, 1), ((3, 4),)), Best Action: RIGHT
State: ((3, 1), ((3, 4),)), Best Action: RIGHT
State: ((4, 1), ((3, 4),)), Best Action: UP
State: ((5, 1), ((3, 4),)), Best Action: UP
State: ((5, 2), ((3, 4),)), Best Action: LEFT
State: ((4, 2), ((3, 4),)), Best Action: UP
State: ((4, 3), ((3, 4),)), Best Action: LEFT
State: ((3, 3), ((3, 4),)), Best Action: LEFT
State: ((4, 4), ((3, 4),)), Best Action: LEFT
State: ((1, 1), ((3, 4),)), Best Action: DOWN
State: ((1, 2), ((3, 4),)), Best Action: DOWN
State: ((3, 2), ((3, 3),)), Best Action: DOWN
State: ((2, 2), ((3, 3),)), Best Action: UP
State: ((2, 1), ((3, 3),)), Best Action: UP
State: ((3, 1), ((3, 3),)), Best Action: UP
State: ((4, 1), ((3, 3),)), Best Action: DOWN
State: ((3, 1), ((3, 2),)), Best Action: UP
State: ((4, 1), ((3, 2),)), Best Action: RIGHT
State: ((4, 2), ((3, 2),)), Best Action: UP
State: ((2, 2), ((3, 2),)), Best Action: UP
State: ((2, 1), ((3, 2),)), Best Action: UP
State: ((5, 1), ((3, 2),)), Best Action: DOWN
State: ((5, 2), ((3, 2),)), Best Action: UP
State: ((3, 3), ((3, 2),)), Best Action: LEFT
State: ((3, 4), ((3, 2),)), Best Action: LEFT
State: ((4, 4), ((3, 2),)), Best Action: UP
State: ((4, 3), ((3, 2),)), Best Action: UP



Improved Policy from Every-Visit Monte Carlo:
State: ((4, 1), ((4, 4),)), Best Action: RIGHT
State: ((3, 1), ((4, 4),)), Best Action: DOWN
State: ((4, 2), ((4, 4),)), Best Action: DOWN
State: ((3, 2), ((4, 4),)), Best Action: RIGHT
State: ((3, 3), ((4, 4),)), Best Action: UP
State: ((3, 4), ((4, 4),)), Best Action: LEFT
State: ((4, 3), ((4, 4),)), Best Action: UP
State: ((2, 2), ((4, 4),)), Best Action: DOWN
State: ((1, 2), ((4, 4),)), Best Action: UP
State: ((1, 1), ((4, 4),)), Best Action: RIGHT
State: ((5, 1), ((4, 4),)), Best Action: RIGHT
State: ((2, 1), ((4, 4),)), Best Action: DOWN
State: ((5, 2), ((4, 4),)), Best Action: DOWN
State: ((4, 2), ((4, 3),)), Best Action: RIGHT
State: ((3, 2), ((4, 3),)), Best Action: RIGHT
State: ((2, 2), ((4, 3),)), Best Action: DOWN
State: ((2, 1), ((4, 3),)), Best Action: DOWN
State: ((3, 3), ((4, 3),)), Best Action: RIGHT
State: ((5, 2), ((4, 3),)), Best Action: UP
State: ((5, 1), ((4, 3),)), Best Action: RIGHT
State: ((4, 1), ((4, 3),)), Best Action: UP
State: ((3, 1), ((4, 3),)), Best Action: RIGHT
State: ((1, 1), ((4, 3),)), Best Action: DOWN
State: ((1, 2), ((4, 3),)), Best Action: DOWN
State: ((3, 4), ((4, 3),)), Best Action: DOWN
State: ((4, 4), ((4, 3),)), Best Action: LEFT
State: ((1, 2), ((5, 1),)), Best Action: UP
State: ((1, 1), ((5, 1),)), Best Action: RIGHT
State: ((2, 2), ((5, 1),)), Best Action: UP
State: ((3, 2), ((5, 1),)), Best Action: RIGHT
State: ((3, 1), ((5, 1),)), Best Action: DOWN
State: ((4, 2), ((5, 1),)), Best Action: LEFT
State: ((5, 2), ((5, 1),)), Best Action: UP
State: ((2, 1), ((5, 1),)), Best Action: UP
State: ((4, 1), ((5, 1),)), Best Action: DOWN
State: ((3, 3), ((5, 1),)), Best Action: UP
State: ((3, 4), ((5, 1),)), Best Action: LEFT
State: ((4, 4), ((5, 1),)), Best Action: UP
State: ((4, 3), ((5, 1),)), Best Action: UP
State: ((3, 1), ((4, 1),)), Best Action: RIGHT
State: ((3, 2), ((4, 1),)), Best Action: DOWN
State: ((3, 3), ((4, 1),)), Best Action: DOWN
State: ((4, 3), ((4, 1),)), Best Action: LEFT
State: ((4, 2), ((4, 1),)), Best Action: DOWN
State: ((4, 4), ((4, 1),)), Best Action: LEFT
State: ((3, 4), ((4, 1),)), Best Action: LEFT
State: ((5, 2), ((4, 1),)), Best Action: LEFT
State: ((4, 3), ((4, 2),)), Best Action: LEFT
State: ((5, 1), ((4, 1),)), Best Action: UP
State: ((2, 2), ((4, 1),)), Best Action: DOWN
State: ((1, 2), ((4, 1),)), Best Action: DOWN
State: ((1, 1), ((4, 1),)), Best Action: DOWN
State: ((2, 1), ((4, 1),)), Best Action: RIGHT
State: ((4, 4), ((4, 2),)), Best Action: LEFT
State: ((3, 4), ((5, 2),)), Best Action: UP
State: ((3, 3), ((5, 2),)), Best Action: RIGHT
State: ((3, 2), ((5, 2),)), Best Action: UP
State: ((4, 3), ((5, 2),)), Best Action: RIGHT
State: ((4, 2), ((5, 2),)), Best Action: DOWN
State: ((4, 1), ((5, 2),)), Best Action: RIGHT
State: ((5, 1), ((5, 2),)), Best Action: UP
State: ((3, 1), ((5, 2),)), Best Action: RIGHT
State: ((4, 4), ((5, 2),)), Best Action: UP
State: ((2, 2), ((5, 2),)), Best Action: RIGHT
State: ((1, 2), ((5, 2),)), Best Action: DOWN
State: ((1, 1), ((5, 2),)), Best Action: DOWN
State: ((2, 1), ((5, 2),)), Best Action: RIGHT
State: ((3, 2), ((4, 2),)), Best Action: RIGHT
State: ((3, 3), ((4, 2),)), Best Action: DOWN
State: ((3, 4), ((4, 2),)), Best Action: DOWN
State: ((4, 1), ((4, 2),)), Best Action: UP
State: ((3, 1), ((4, 2),)), Best Action: RIGHT
State: ((2, 1), ((4, 2),)), Best Action: RIGHT
State: ((2, 2), ((4, 2),)), Best Action: RIGHT
State: ((1, 2), ((4, 2),)), Best Action: DOWN
State: ((1, 1), ((4, 2),)), Best Action: RIGHT
State: ((5, 1), ((4, 2),)), Best Action: UP
State: ((5, 2), ((4, 2),)), Best Action: DOWN
State: ((4, 1), ((1, 2),)), Best Action: DOWN
State: ((5, 1), ((1, 2),)), Best Action: DOWN
State: ((5, 2), ((1, 2),)), Best Action: LEFT
State: ((4, 2), ((1, 2),)), Best Action: RIGHT
State: ((4, 3), ((1, 2),)), Best Action: DOWN
State: ((4, 4), ((1, 2),)), Best Action: LEFT
State: ((3, 3), ((1, 2),)), Best Action: DOWN
State: ((3, 2), ((1, 2),)), Best Action: RIGHT
State: ((3, 1), ((1, 2),)), Best Action: DOWN
State: ((3, 4), ((1, 2),)), Best Action: DOWN
State: ((2, 1), ((1, 2),)), Best Action: UP
State: ((1, 1), ((1, 2),)), Best Action: UP
State: ((2, 2), ((1, 2),)), Best Action: DOWN
State: ((3, 2), ((2, 2),)), Best Action: UP
State: ((3, 1), ((2, 2),)), Best Action: UP
State: ((4, 2), ((2, 2),)), Best Action: DOWN
State: ((2, 1), ((2, 2),)), Best Action: UP
State: ((3, 3), ((2, 2),)), Best Action: UP
State: ((3, 4), ((2, 2),)), Best Action: UP
State: ((4, 4), ((2, 2),)), Best Action: UP
State: ((4, 3), ((2, 2),)), Best Action: UP
State: ((4, 1), ((2, 2),)), Best Action: DOWN
State: ((5, 1), ((2, 2),)), Best Action: UP
State: ((5, 2), ((2, 2),)), Best Action: UP
State: ((4, 2), ((3, 2),)), Best Action: RIGHT
State: ((5, 2), ((3, 2),)), Best Action: UP
State: ((5, 1), ((3, 2),)), Best Action: UP
State: ((4, 1), ((3, 2),)), Best Action: UP
State: ((4, 3), ((3, 2),)), Best Action: RIGHT
State: ((3, 1), ((3, 2),)), Best Action: RIGHT
State: ((3, 3), ((3, 2),)), Best Action: LEFT
State: ((2, 1), ((3, 2),)), Best Action: DOWN
State: ((1, 1), ((3, 2),)), Best Action: DOWN
State: ((1, 2), ((3, 2),)), Best Action: UP
State: ((2, 2), ((3, 2),)), Best Action: DOWN
```

Value iteration is faster than monte carlo. In Monte Carlo approach, the episode needs to be stopped every time the agent 
is stuck.