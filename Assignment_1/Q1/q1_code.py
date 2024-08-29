import numpy as np 

# Policy Evaluation 
def policy_eval(state):
    # repeat until convergence
    iteration = 0
    while True:
        delta = 0
        new_V = V
        
        # calculating best policy for curent state    
        for s in states:
            action = policy[s]
            new_V[s] = rewards[s] + gamma * sum([transitions[s][action][s_next] * V[s_next] for s_next in states])
            delta = max(delta, np.abs(new_V[s] - V[s]))
                                          
        # update iteration no.
        iteration += 1
        
        # check for maximum iterations
        if iteration >= max_iteration:
            break
        
        # check for convergence
        if np.max(np.abs(new_V - V)) < delta:
            break

# Policy improvement
def policy_improve():
    policy_stable = True
    
    for s in states:
        old_action = policy[s]
        
        # calculate values of all actions from current state
        q=[]
        
        for a in actions:
            q_value = sum([transitions[s][a][s_prime] * V[s_prime] for s_prime in states]) 
            q.append(q_value)
        
        # calculating maximum values for the state s given action a
        for s in states:
            policy[s] = np.argmax([rewards[s] + gamma * q[a] for a in actions])
        
        if policy[s] != old_action :
            policy_stable = False
        
    return policy_stable

# Policy iteration 
def policy_iteration():
    iteration = 0
    while iteration < max_iteration:
        policy_eval(policy)

        if policy_improve():
            break

        iteration += 1

    return policy, V

# User defined functions------------------------------------------------------------

# States
states = [0, 1, 2] 
# Actions
actions = [0, 1] 
# Reward function 
rewards = [-1, 3, 1]
# State transiton functions
transitions = np.array([
    [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],  #state 0
    [[0.0, 0.7, 0.3], [0.0, 0.2, 0.8]],  #state 1
    [[0.3, 0.6, 0.1], [0.0, 0.0, 1.0]]   #state 2
])

# Initiation Step------------------------------------------------------------------------------------
gamma  = 0.79 #discount factor
thresold = 10e-6 #threshold for convergence
max_iteration = 100 #maximum number of iterations

# setting policy values
policy = np.zeros(len(states),dtype=int)

# setting value function values
V = np.zeros(len(states))

# Run Policy Iteration-----------------------------------------------------------------------------
optimal_policy, optimal_values = policy_iteration()

# Print results
print("state 0: Hostel\t state 1: AcadmicBuilding \t state 2: Canteen")
print('action 0 : study\t action 1: Eat')
print("-----------------------------OPTIMAL_POLICY-----------------------------")
print(optimal_policy)
print("-----------------------------OPTIMAL_VALUES-----------------------------")
print(optimal_values)