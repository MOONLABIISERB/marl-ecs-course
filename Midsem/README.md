## MidSem
-by Vasu Dhull 21302
## Setting up the Virtual Environment

To set up a Python virtual environment and install the required dependencies, follow these steps:

1. ```bash
   python -m venv myenv
   ```

2. ```bash
     myenv\Scripts\activate
     ```

3. ```bash
   pip install -r requirements.txt
   ```

## Q-Learning Implementation
Q-learning is implemented by maintaining a Q-table (self.q_table) that stores the estimated value of taking an action in a given state. For each episode, the agent selects an action using the choose_action method, which picks the action with the highest Q-value for the current state (exploiting the Q-table). After executing the action, the agent receives a reward and updates the Q-value using the update_q_value method, based on the temporal difference error: the difference between the expected future reward (reward + gamma * max Q(next_state)) and the current Q-value. This update is controlled by the learning rate (alpha) and discount factor (gamma), guiding the agent toward maximizing long-term rewards.

## Q-Learning Parameters

| Parameter     | Value  | Description              |
|---------------|--------|--------------------------|
| `self.alpha`  | 0.1    | Learning rate (α)         |
| `self.gamma`  | 0.999  | Discount factor (γ)       |

## Results
For 5k episodes :

![Rew vs Ep 5k](https://github.com/user-attachments/assets/a39620c8-443b-426e-ae8d-5f49504e206e)

For 10k episodes :

![Rew vs Ep 10k](https://github.com/user-attachments/assets/80518589-c9f4-4406-b8bb-79cc34dae849)



