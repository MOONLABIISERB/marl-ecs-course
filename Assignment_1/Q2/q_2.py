import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Defining all given parameters
grid_sz = (9, 9) #grid size
goal_pos = (8, 8) #goal position

# Calculating all the states coordinates
states = []
for s in np.ndindex(grid_sz):
  states.append(s)

# Coordinate of all the walls
walls = [(3,1), (3,2), (3,3), (1,3), (2,3), (5,5), (6,5), (7,5), (5,8), (5,6), (5,7), (8,5)]

# List of all actions
actions = ['stay', 'up', 'down', 'left', 'right']

#Assign initial values for policy 
policy = {}
for i in range(grid_sz[0]):
    for j in range(grid_sz[1]):
        policy[(i, j)] = actions[0]

gamma = 0.9 #Discount factor
theta = 10e-6 #Threshold value

# Rewards for each state
rewards = np.zeros((grid_sz[0],grid_sz[1]))
rewards[goal_pos] = 1

# assigning to all states
value = np.zeros(grid_sz)



class ValueIteration:
    def __init__(self, grid_sz, goal_pos, policy, states, actions, walls, value, rewards, transition, gamma, theta):
        self.value = value
        self.policy = policy        
        
    def value_iteration(self):
        while True:
            delta = 0
            for s in states:
                if s == goal_pos:
                    policy[s]='stay'
                    break                
                v = self.value[s]
                max_value = float('-inf')                
                for a in actions:
                    next_s = transition(s,a)
                    v_new = rewards[next_s] + gamma*self.value[next_s]
                    max_value = max(v_new,max_value)                
                self.value[s] = max_value
                delta = max(delta, np.abs(self.value[s] - v))                
            if delta < theta: # when the threshold for error is reached
                break
        
        for s in states:
            action_values = {}
            for a in actions:
                next_state = transition(s, a)
                value = rewards[next_state] + gamma*self.value[next_state]
                action_values[a] = value
            self.policy[s] = max(action_values, key=action_values.get)
        
        return self.value, self.policy             


class PolicyIteration:
    def __init__(self, grid_sz, goal_pos, policy, states, actions, walls, value, rewards, transition, gamma, theta):
        self.policy = policy
        self.value = value        
    def policy_eval(self):
        while True:
            delta = 0
            for s in states:
                if s==goal_pos:
                    continue
            
                v = self.value[s]
                a = self.policy[s]
                s_next = transition(s, a)
                self.value[s] = rewards[s_next] + gamma*self.value[s_next]    
                delta = max(delta,abs(v-self.value[s]))
            
            #to check convergence of the value
            if delta < theta:
                break  
        
    def policy_improvement(self):
        policy_stable = True
    
        for s in states:
            if s==goal_pos:
                continue
            
            old_action = self.policy[s]
            action_values = {}
            
            for a in actions:
                s_next = transition(s, a)
                action_values[a] = rewards[s_next] + gamma*self.value[s_next]
                
            best_action = max(action_values, key=action_values.get)
            self.policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
        
    def policy_iteration(self):
        while True:
            self.policy_eval()
            if self.policy_improvement():
                break
        
        return self.value, self.policy
        
    
def transition(state, action):
    walls = [(3,1), (3,2), (3,3), (1,3), (2,3), (5,5), (6,5), (7,5), (5,8), (5,6), (5,7), (8,5)] 
    grid_size =(9,9)
    x, y = state
    
    
    
    if action == 'up':
        next_state = (max(x - 1, 0), y)
    elif action == 'down':
        next_state = (min(x + 1, grid_size[0] - 1), y)
    elif action == 'left':
        next_state = (x, max(y - 1, 0))
    elif action == 'right':
        next_state = (x, min(y + 1, grid_size[1] - 1))
    elif action == 'stay':
        next_state = state
        
    if next_state == (8,8):
        return (8,8)

    if next_state == (2, 2):
        return (6,6)   
    
    if next_state in walls:
        return state

    return next_state
  
def plot(policy, grid_size, walls, name) -> None:
    
    X, Y = np.meshgrid(range(grid_size[0]), range(grid_size[1]))
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    
    action_vectors = {
        'up': (0, -1), 'down': (0, 1),
        'left': (-1, 0), 'right': (1, 0),
        'stay': (0, 0)
    }
    
    for state, action in policy.items():
        if state not in walls: 
            dx, dy = action_vectors[action]
            U[state] = dx
            V[state] = dy
   
    plt.figure()
    plt.title(name)
    plt.grid( color='r', linestyle='-.', linewidth='0.2')
    plt.quiver(X, Y, U, V, pivot='middle', color='k', width=0.005, headlength=4)
    
    plt.plot(0, 0, 'c*', markersize = '25', alpha = 0.5) #Start
    plt.plot(2, 2, 'ro', markersize = '15', alpha = 0.5) #Tunnel inlet
    plt.plot(6, 6, 'bo', markersize = '15', alpha = 0.5) #Tunnel outlet
    plt.plot(8, 8, 'g*', markersize = '25', alpha = 0.5) #End goal
    
    # Walls
    for wall in walls:
        plt.plot(wall[1], wall[0],'x', color = 'y', markersize ='10')
    
    plt.gca()
    plt.show()

def main():
    np.printoptions(precision=3)
    print("-----------------------------------------------------------------VALUE ITERATION VALUES--------------------------------------------------------------------------------")
    value_iteratn = ValueIteration(grid_sz, goal_pos, policy, states, actions, walls, value, rewards, transition, gamma, theta)
    value_vi,policy_vi = value_iteratn.value_iteration()
    plot(policy_vi, grid_sz, walls, "VALUE ITERATION")
    print(f"------Transition Value: {value_vi}")
    print(f"------Policies: {policy_vi}")


    print("-----------------------------------------------------------------POLICY ITERATION VALUES--------------------------------------------------------------------------------")
    policy_iteratn = PolicyIteration(grid_sz, goal_pos, policy, states, actions, walls, value, rewards, transition, gamma, theta)
    value_pi, policy_pi = policy_iteratn.policy_iteration()
    plot(policy_pi, grid_sz, walls, "POLICY ITERATION")
    print(f"------Transition Value: {value_pi}")
    print(f"------Policies: {policy_pi}")

if __name__ == "__main__":
    main()


