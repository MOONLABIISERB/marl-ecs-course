from modified_tsp import ModTSP
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
def state_to_tuple(state):
    return tuple(state.astype(int))

class QLearningAgent:
    def __init__(self, action_space = 10, learning_rate=0.1, discount_factor=0.99):
        self.action_space = action_space 
        self.learning_rate = learning_rate 
        self.discount_factor = discount_factor  
        self.q_table = {}  
        self.td_errors = [] 

    def choose_action(self, state):
        """Choose the best action based on the current Q-values (greedy policy)."""
        state_str = state_to_tuple(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(self.action_space)  
        return np.argmax(self.q_table[state_str])  

    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value for the given state-action pair."""
        state_str = state_to_tuple(state)
        next_state_str = state_to_tuple(next_state)

        
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(self.action_space)


        next_action = np.argmax(self.q_table[next_state_str]) 
        td_target = reward + self.discount_factor * self.q_table[next_state_str][next_action]
        td_error = td_target - self.q_table[state_str][action]

        self.td_errors.append(abs(td_error))

        
        self.q_table[state_str][action] += self.learning_rate * td_error


def main() -> None:
    """Main function to run Q-learning agent in Modified TSP environment."""

    num_episodes = 9999

    env = ModTSP()
    agent = QLearningAgent()
    
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'logs/{current_time}'
    writer = SummaryWriter(log_dir)
    pbar = tqdm(range(num_episodes), desc="Training Progress")
    for ep in pbar:
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated

            agent.update_q_value(state, action, reward, next_state) 

            total_reward += reward
            state = next_state
        writer.add_scalar('Reward/Episode', total_reward, ep)
        writer.add_scalar('Loss/Episode', np.mean(agent.td_errors), ep)

    state, _ = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action) 
        print('action', action)
        done = terminated or truncated
        state = next_state

if __name__ == "__main__":
    main()