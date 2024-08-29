import numpy as np


class ValueIteration:
    def __init__(self, states, actions, transition_probabilities, rewards, discount_factor, theta) -> None:
        '''
        Value Iteration Constructor

        Args:
        states: list, states
        actions: list, actions
        transition_probabilities: dict, transition probabilities
        rewards: dict, rewards
        discount_factor: float, discount factor
        theta: float, threshold

        Returns:
        None
        '''

        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.discount_factor = discount_factor
        self.theta = theta



    def value_iteration(self) -> tuple:
        '''
        Value Iteration Algorithm

        Returns:
        V: dict, value function
        policy: dict, optimal policy
        '''

        V = {s: 0 for s in self.states}
        policy = {s: None for s in self.states}

        while True:
            delta = 0
            for s in self.states:
                v = V[s]
                max_value = float('-inf')
                for a in self.actions:
                    action_value = 0
                    for next_state in self.states:
                        prob = self.transition_probabilities.get((s, a, next_state), 0)
                        reward = self.rewards.get((s, a, next_state), 0)
                        action_value += prob * (reward + self.discount_factor * V[next_state])
                    max_value = max(max_value, action_value)
                V[s] = max_value
                delta = max(delta, abs(v - V[s]))

            if delta < self.theta:
                break

        self.derive_policy(policy, V)
        return V, policy


    


    def derive_policy(self, policy, V) -> None:
        '''
        Derive optimal policy from value function

        Args:
        policy: dict, policy
        V: dict, value function

        Returns:
        None
        '''

        for s in self.states:
            action_values = {}
            for a in self.actions:
                action_value = 0
                for next_state in self.states:
                    prob = self.transition_probabilities.get((s, a, next_state), 0)
                    reward = self.rewards.get((s, a, next_state), 0)
                    action_value += prob * (reward + self.discount_factor * V[next_state])
                action_values[a] = action_value
            policy[s] = max(action_values, key=action_values.get)
    


def main():
    states = ['s1', 's2', 's3'] #s1=hostel, s2=academic building, s3=canteen
    actions = ['a1', 'a2'] #a1=attend class, a2=eat food

    transition_probabilities = {
    ('s1', 'a1', 's2'): 0.5, 
    ('s1', 'a1', 's1'): 0.5,
    ('s1', 'a2', 's3'): 1.0,
    ('s2', 'a1', 's2'): 0.7, 
    ('s2', 'a1', 's3'): 0.3,
    ('s2', 'a2', 's3'): 0.8,
    ('s2', 'a2', 's2'): 0.2,
    ('s3', 'a1', 's1'): 0.6,
    ('s3', 'a1', 's2'): 0.3,
    ('s3', 'a1', 's1'): 0.1,
    ('s3', 'a2', 's3'): 1
    }

    rewards = {
    ('s1', 'a1', 's2'): 3, 
    ('s1', 'a1', 's1'): -1,
    ('s1', 'a2', 's3'): 1,
    ('s2', 'a1', 's2'): 3, 
    ('s2', 'a1', 's3'): 1,
    ('s2', 'a2', 's3'): 1,
    ('s2', 'a2', 's2'): 3,
    ('s3', 'a1', 's1'): 3,
    ('s3', 'a1', 's2'): -1,
    ('s3', 'a1', 's1'): 1,
    ('s3', 'a2', 's3'): 1
    }

    discount_factor = 0.9
    theta = 0.01

    v_it = ValueIteration(states, actions, transition_probabilities, rewards, discount_factor, theta)
    V, p = v_it.value_iteration()

    print("V: ", V)
    print("Optimal Policy: ", p)


if __name__ == "__main__":
    main()