## How to run code:
#### Run train.py. This trains the Q-learning algorithm. On running it, a file q_table.npy will appear in the same folder.
#### Run test.py. This will load q_table.npy automatically and will print rewards for each episode.


## train.py explaination
#### ----The state consists of the current location and a binary vector indicating whether each target has been visited.
#### ----If simply current location as a state in the Q table then it would not be a Markov Decision Process. We need to include the previously visited states as part of the Q table. Therefore, the Q table is created as a dictionary where each key is a state-action pair, and the value is the Q-value (initialized to 0). So the total length of this dictionary is 102400.
#### ----get_max: Finds the maximum Q-value for a given state.
#### ----best_action: Returns the action with the highest Q-value for a given state.
#### ----epsilon_greedy: Implements the epsilon-greedy policy where the agent either explores a random action (with probability epsilon) or exploits the best-known action.
#### ----After training, it saves the Q-table to a file and plots the rewards per episode and the running average reward.
