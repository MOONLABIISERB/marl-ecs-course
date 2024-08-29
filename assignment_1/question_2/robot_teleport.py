import numpy as np
import matplotlib.pyplot as plt


class ValueIteration:
    def __init__(self, grid_size, goal_position, actions, transition, rewards, discount_factor, theta):
        self.grid_size = grid_size
        self.goal_position = goal_position
        self.actions = actions
        self.transition = transition
        self.rewards = rewards
        self.discount_factor = discount_factor
        self.theta = theta
        self.value_function = np.zeros(grid_size)
        self.policy = {state: np.random.choice(actions) for state in np.ndindex(grid_size)}
    
    def value_iteration(self) -> tuple:
        '''
        Value Iteration Algorithm

        Returns:
        value_function: dict, value function
        policy: dict, optimal policy
        '''

        while True:
            delta = 0
            for state in np.ndindex(self.grid_size):
                if state == self.goal_position:
                    continue 
                v = self.value_function[state]
                max_value = float('-inf')
                for action in self.actions:
                    next_state = self.transition(state, action)
                    reward = self.rewards[next_state]
                    value = reward + self.discount_factor * self.value_function[next_state]
                    max_value = max(max_value, value)
                self.value_function[state] = max_value
                delta = max(delta, abs(v - self.value_function[state]))
            
            if delta < self.theta:
                break
        
        for state in np.ndindex(self.grid_size):
            action_values = {}
            for action in self.actions:
                next_state = self.transition(state, action)
                reward = self.rewards[next_state]
                value = reward + self.discount_factor * self.value_function[next_state]
                action_values[action] = value
            self.policy[state] = max(action_values, key=action_values.get)
        
        return self.value_function, self.policy



class PolicyIteration:
    def __init__(self, grid_size, goal_position, actions, transition, rewards, discount_factor, theta):
        self.grid_size = grid_size
        self.goal_position = goal_position
        self.actions = actions
        self.transition = transition
        self.rewards = rewards
        self.discount_factor = discount_factor
        self.theta = theta
        self.value_function = np.zeros(grid_size)
        self.policy = {state: np.random.choice(actions) for state in np.ndindex(grid_size)}
    

    def policy_evaluation(self) -> None:
        '''
        Policy Evaluation

        Returns:
        None
        '''

        while True:
            delta = 0
            for state in np.ndindex(self.grid_size):
                if state == self.goal_position:
                    continue  # Skip goal state
                
                v = self.value_function[state]
                action = self.policy[state]
                next_state = self.transition(state, action)
                reward = self.rewards[next_state]
                self.value_function[state] = reward + self.discount_factor * self.value_function[next_state]
                delta = max(delta, abs(v - self.value_function[state]))
            
            if delta < self.theta:
                break
    


    def policy_improvement(self) -> bool:
        '''
        Policy Improvement

        Returns:
        policy_stable: bool, policy stable or not
        '''

        policy_stable = True
        for state in np.ndindex(self.grid_size):
            if state == self.goal_position:
                continue  # Skip goal state
            
            old_action = self.policy[state]
            action_values = {}
            for action in self.actions:
                next_state = self.transition(state, action)
                reward = self.rewards[next_state]
                value = reward + self.discount_factor * self.value_function[next_state]
                action_values[action] = value
            best_action = max(action_values, key=action_values.get)
            self.policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    


    def policy_iteration(self) -> tuple:
        '''
        Policy Iteration Algorithm

        Returns:
        value_function: dict, value function
        policy: dict, optimal policy
        '''

        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break
        
        return self.value_function, self.policy



def transition(state, action) -> tuple:
    '''
    Transition function

    Args:
    state: tuple, current state
    action: str, action

    Returns:
    next_state: tuple, next state
    '''

    tunnel_in = (2, 2)
    tunnel_out = (6, 6)
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
    elif action == 'up-left':
        next_state = (max(x - 1, 0), max(y - 1, 0))
    elif action == 'up-right':
        next_state = (max(x - 1, 0), min(y + 1, grid_size[1] - 1))
    elif action == 'down-left':
        next_state = (min(x + 1, grid_size[0] - 1), max(y - 1, 0))
    elif action == 'down-right':
        next_state = (min(x + 1, grid_size[0] - 1), min(y + 1, grid_size[1] - 1))
    elif action == 'stay':
        next_state = state

    if next_state == tunnel_in:
        next_state = tunnel_out
    

    if next_state in walls:
        next_state = state
    
    return next_state



def plot_policy(policy, grid_size, walls) -> None:
    '''
    Plot policy

    Args:
    policy: dict, policy
    grid_size: tuple, grid size
    walls: list, wall coordinates

    Returns:
    None
    '''

    X, Y = np.meshgrid(range(grid_size[1]), range(grid_size[0]))
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    
    action_vectors = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0),
        'up-left': (-1, -1),
        'up-right': (1, -1),
        'down-left': (-1, 1),
        'down-right': (1, 1),
        'stay': (0, 0)
    }
    
    for state, action in policy.items():
        if state not in walls: 
            dx, dy = action_vectors[action]
            U[state] = dx
            V[state] = dy
   
    plt.figure()
    plt.grid(visible=True, color='b', linestyle='-', linewidth=0.5)
    plt.quiver(X, Y, U, V)
    
    for wall in walls:
        plt.plot(wall[1], wall[0], 'ro')
    
    plt.gca()
    plt.show()



def main():
    grid_size = (9, 9)
    goal_position = (8, 8) 
    rewards = np.zeros(grid_size)
    rewards[goal_position] = 1 

    walls = [(3,1), (3,2), (3,3), (1,3), (2,3), (5,5), (6,5), (7,5), (5,8), (5,6), (5,7), (8,5)]
    actions = ['up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right', 'stay']

    policy = {}
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            policy[(i, j)] = np.random.choice(actions)

    print("VALUE ITERATION:")
    v_it = ValueIteration(grid_size, goal_position, actions, transition, rewards, discount_factor=0.9, theta=0.01)
    _, policy_v_it = v_it.value_iteration()
    print(_)
    print(policy_v_it)
    plot_policy(policy_v_it, grid_size, walls)
    # plt.savefig('value_iteration.png')  

    print('\n')
    print("---------------------------------")
    print('\n')

    print("POLICY ITERATION:")
    p_it = PolicyIteration(grid_size, goal_position, actions, transition, rewards, discount_factor=0.9, theta=0.01)
    _, policy_p_it = p_it.policy_iteration()
    print(_)
    print(policy_p_it)
    plot_policy(policy_p_it, grid_size, walls)
    # plt.savefig('policy_iteration.png')





if __name__ == "__main__":
    main()
