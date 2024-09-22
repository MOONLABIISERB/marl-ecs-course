import numpy as np  
import matplotlib.pyplot as plt

  
# Define the grid dimensions  
GRID_WIDTH = 7  
GRID_HEIGHT = 6  
  
# Define the state space  
STATE_SPACE = [(i, j) for i in range(GRID_HEIGHT) for j in range(GRID_WIDTH)]  
  
# Define the action space  
ACTION_SPACE = ['UP', 'DOWN', 'LEFT', 'RIGHT']  
  
# Define the reward structure  
REWARD_STRUCTURE = {  
   'box_not_placed': -1,  
   'box_placed': 0  
}  
  
# Define the termination conditions  
TERMINATION_CONDITIONS = {  
   'all_boxes_placed': True,  
   'box_stuck': True  
}  
  
# Define the storage locations  
STORAGE_LOCATIONS = [(1, 1), (2, 2), (3, 3)]  # example storage locations  
  
# Initialize the value function  
V = {s: 0 for s in STATE_SPACE}  
  
# Function to get the next state and reward  
def get_next_state_and_reward(s, a):  
   # Get the current state and action  
   x, y = s  
   # Get the next state  
   if a == 'UP':  
      x -= 1  
   elif a == 'DOWN':  
      x += 1  
   elif a == 'LEFT':  
      y -= 1  
   elif a == 'RIGHT':  
      y += 1  
   # Check if the next state is valid  
   if (x, y) in STATE_SPACE:  
      # Get the reward  
      if (x, y) in STORAGE_LOCATIONS:  
        reward = REWARD_STRUCTURE['box_placed']  
      else:  
        reward = REWARD_STRUCTURE['box_not_placed']  
      return (x, y), reward  
   else:  
      return s, REWARD_STRUCTURE['box_not_placed']  
  
# Function to get the expected value  
def get_expected_value(s, a):  
   # Get the next state and reward  
   s_prime, r = get_next_state_and_reward(s, a)  
   # Compute the expected value  
   return r + 0.9 * V[s_prime]  # discount factor 0.9  
  
# Value iteration algorithm  
for _ in range(1000):  # maximum number of iterations  
   for s in STATE_SPACE:  
      # Compute the expected value of taking each possible action  
      expected_values = []  
      for a in ACTION_SPACE:  
        # Get the next state s' and reward r  
        s_prime, r = get_next_state_and_reward(s, a)  
        # Compute the expected value  
        expected_value = r + 0.9 * V[s_prime]  # discount factor 0.9  
        expected_values.append(expected_value)  
      # Update the value function  
      V[s] = max(expected_values)  
  
# Get the optimal policy  
policy = {}  
for s in STATE_SPACE:  
   # Get the action with the maximum expected value  
   a = np.argmax([get_expected_value(s, a) for a in ACTION_SPACE])  
   policy[s] = ACTION_SPACE[a]

   # Define the boxes  
BOXES = [(0, 0), (1, 2), (2, 1)]  # example boxes  
  
# Define the warehouse agent  
AGENT = (3, 2)  # example agent location  
  
# Create a 2D array to represent the grid  
grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))  
  
# Mark the storage locations  
for loc in STORAGE_LOCATIONS:  
   grid[loc[0], loc[1]] = 1  
  
# Mark the boxes  
for box in BOXES:  
   grid[box[0], box[1]] = 2  
  
# Mark the agent  
grid[AGENT[0], AGENT[1]] = 3  
  
# Create a color map  
cmap = plt.get_cmap('viridis')  
  
# Plot the grid  
plt.imshow(grid, cmap=cmap)  
plt.title('Output Grid')  
plt.xlabel('Column')  
plt.ylabel('Row')  
plt.show()