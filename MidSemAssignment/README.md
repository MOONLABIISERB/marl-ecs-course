## How to run code:
##### Run train.py. This trains the Q-learning algorithm. On running it, a file q_table.npy will appear in the same folder.
##### Run test.py. This will load q_table.npy automatically and will print rewards for each episode.


## train.py explanation
 ##### ----Algorithm Used: Q-learning Algorithm
 ##### ----The state consists of the current location and a binary vector indicating whether each target has been visited.
 
 ##### ----If simply current location as a state in the Q table then it would not be a Markov Decision Process. We need to include the previously visited states as part of the Q table. Therefore, the Q table is created as a dictionary where each key is a state-action pair, and the value is the Q-value (initialized to 0). So the total length of this dictionary is 102400.  
 ##### An example of a state, action key for Q table would be ([5, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], 7). This indicates that current state is 5 and states 1 and 5 have been visited as indicated by the 1's. And the next action is 7.
 
 ##### ----get_max: Finds the maximum Q-value for a given state.
 
 ##### ----best_action: Returns the action with the highest Q-value for a given state.
 
 ##### ----epsilon_greedy: Implements the epsilon-greedy policy where the agent either explores a random action (with probability epsilon) or exploits the best-known action.
 
 ##### ----After training, it saves the Q-table to a file and plots the rewards per episode and the running average reward.

#### Output Snippet
``` bash
Episode 39995 : 213.45571899414062  epsilon: 0.0010000000003462382
Visited Path: [8, 0, 3, 9, 2, 7, 6, 1, 4, 5]
Unique points: 10
Episode 39996 : 200.5977602005005  epsilon: 0.001000000000345996
Visited Path: [3, 8, 2, 9, 6, 7, 4, 0, 5, 1]
Unique points: 10
Episode 39997 : 213.45571899414062  epsilon: 0.0010000000003457538
Visited Path: [8, 0, 3, 9, 2, 7, 6, 1, 4, 5]
Unique points: 10
Episode 39998 : 57.16754722595215  epsilon: 0.001000000000345512
Visited Path: [7, 3, 6, 9, 2, 8, 4, 0, 5, 1]
Unique points: 10
Episode 39999 : 171.42115020751953  epsilon: 0.0010000000003452703
Visited Path: [1, 6, 8, 3, 9, 7, 2, 4, 0, 5]
Unique points: 10
```

#### Plot

<img width="935" alt="RewardVsEpisodes" src="https://github.com/user-attachments/assets/dcde9bb8-6276-488d-9f48-985f77d65f97">


## test.py Explanation:
#### Testing (main):
##### ----Loads the Q-table trained in train.py and runs test episodes to evaluate the learned policy.
##### ----During each episode, it selects the best action based on the Q-table and calculates the cumulative reward.
##### ----After testing, it identifies the path that gave the maximum reward and visualizes this best path on a 2D plot.

#### Output Snippet:
``` bash
Test Episode 95: Total Reward: 33.77323532104492   Visited Path: [7, 6, 3, 9, 2, 8, 4, 0, 5, 1]
Test Episode 96: Total Reward: 215.94062328338623   Visited Path: [3, 2, 7, 6, 9, 8, 4, 0, 5, 1]
Test Episode 97: Total Reward: 142.2993984222412   Visited Path: [1, 3, 0, 9, 6, 7, 2, 8, 4, 5]
Test Episode 98: Total Reward: 158.78661060333252   Visited Path: [5, 0, 7, 9, 3, 8, 2, 6, 4, 1]
Test Episode 99: Total Reward: 158.78661060333252   Visited Path: [5, 0, 7, 9, 3, 8, 2, 6, 4, 1]
Average Reward over 100 test episodes: 151.21117995738985
```

#### Scatter Plot:

<img width="868" alt="IdealPath" src="https://github.com/user-attachments/assets/dc046f80-e314-413e-93e9-0b447a599949">


