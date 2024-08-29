import numpy as np


class PolicyIteration:
    def __init__(self, states, actions, transition_probabilities, rewards, discount_factor, theta) -> None:
        '''
        Policy Iteration Constructor

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



    def policy_iteration(self)-> tuple:
        '''
        Policy Iteration Algorithm

        Returns:
        V: dict, value function
        policy: dict, optimal policy
        '''

        V = {s: 0 for s in self.states}
        policy = {s: self.actions[0] for s in self.states}

        while True:

            # Policy Evaluation
            while True:
                delta = 0
                for s in self.states:
                    v = V[s]
                    action = policy[s]
                    V[s] = sum(self.transition_probabilities.get((s, action, next_state), 0) *
                                    (self.rewards.get((s, action, next_state), 0) + 
                                     self.discount_factor * V[next_state])
                                    for next_state in self.states)
                    delta = max(delta, abs(v - V[s]))
                if delta < self.theta:
                    break

            # Policy Improvement
            policy_stable = True
            for s in self.states:
                old_action = policy[s]
                action_values = {}
                for a in self.actions:
                    action_values[a] = sum(self.transition_probabilities.get((s, a, next_state), 0) *
                                           (self.rewards.get((s, a, next_state), 0) + 
                                            self.discount_factor * V[next_state])
                                           for next_state in self.states)
                policy[s] = max(action_values, key=action_values.get)

                if old_action != policy[s]:
                    policy_stable = False

            if policy_stable:
                break

        return V, policy  



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

    p_it = PolicyIteration(states, actions, transition_probabilities, rewards, discount_factor, theta)
    V, p = p_it.policy_iteration()

    print("Optimal Policy: ", p)
    print("V: ", V)



if __name__ == '__main__':
    main()