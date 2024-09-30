import random
import numpy as np
# from modified_tsp import ModTSP

class SARSA():
    def __init__(self, states, actions, rewards, distances, steps = 9, n_episodes = 1000):
        self.states = states
        length = len(self.states)
        self.actions = actions
        self.rewards = rewards
        self.distances = distances
        self.steps = steps
        self.n_episodes = n_episodes
        self.Q = np.zeros((length,length), dtype =float)


    def training(self,state):
        self.Q = np.zeros((10,10), dtype =float)
        episodic_rewards =[]
        cumilative_rewards = []

        for ep in range(self.n_episodes):
            
            # state = np.random.choice(self.states)
            distance_traveled = 0
            visited_states = []
            visited_states.append(state)
            total_rewards = 0
                
            for i in range(self.steps):

            # print(state)
                reward = self.rewards[state]

                next_state, temp_reward = self.q_value(state, reward, distance_traveled, visited_states)
                
                total_rewards += temp_reward

                visited_states.append(next_state)
                distance_traveled += self.distances[state,next_state]
                state = next_state    
                            
            episodic_rewards.append(total_rewards)
            cumilative_rewards.append(np.mean(episodic_rewards))
        
        return self.Q, cumilative_rewards, episodic_rewards
    
    def q_value(self, state, rewards, dist, visited_states):
        alpha = 0.001
        gamma = 0.9
        epsilon = 0.1

        for action in self.actions:

            reward = rewards
            next_state = action
            distance = dist + self.distances[state,action]

            if next_state not in visited_states:
                reward = float(reward - distance)

            else:
                reward = float( -1e4 - distance)
            
            current_q = self.Q[state][action]
            next_q = max(self.Q[next_state])

            self.Q[state][action] = current_q + alpha * ( reward + gamma *(next_q) - current_q)

        if random.random() > epsilon:
            next_state = np.argmax(self.Q[state])
        else:
            next_state = np.random.choice(self.actions)
        # next_state = np.argmax(self.Q[state])

        return next_state,reward
    