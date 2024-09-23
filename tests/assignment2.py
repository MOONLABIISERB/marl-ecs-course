from typing import Dict, List, Tuple, Callable
import numpy as np
from abc import ABC, abstractmethod


class State:
    """
    Represents a state in the Markov Decision Process.

    Attributes:
        id (int): A unique identifier for the state.
        name (str): A descriptive name for the state.
    """

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name


class Action:
    """
    Represents an action that can be taken in the Markov Decision Process.

    Attributes:
        id (int): A unique identifier for the action.
        name (str): A descriptive name for the action.
    """

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name


class Policy:
    """
    Represents a policy for the Markov Decision Process.

    A policy is a mapping from states to actions.
    """

    def __init__(self, state_action_map: Dict[State, Action]):
        self.state_action_map = state_action_map

    def get_action(self, state: State) -> Action:
        """
        Get the action for a given state according to this policy.

        Args:
            state (State): The current state.

        Returns:
            Action: The action to take in the given state.
        """
        return self.state_action_map[state]


class Environment(ABC):
    """
    Abstract base class for an environment in the Markov Decision Process.

    This class defines the interface that all environments must implement.
    """

    @abstractmethod
    def get_states(self) -> List[State]:
        """
        Get all possible states in the environment.

        Returns:
            List[State]: A list of all possible states.
        """
        pass

    @abstractmethod
    def get_actions(self, state: State) -> List[Action]:
        """
        Get all possible actions for a given state.

        Args:
            state (State): The current state.

        Returns:
            List[Action]: A list of all possible actions in the given state.
        """
        pass

    @abstractmethod
    def get_transition_prob(
        self, state: State, action: Action, next_state: State
    ) -> float:
        """
        Get the transition probability from one state to another given an action.

        Args:
            state (State): The current state.
            action (Action): The action taken.
            next_state (State): The resulting state.

        Returns:
            float: The probability of transitioning from state to next_state given action.
        """
        pass

    @abstractmethod
    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        """
        Get the reward for a state-action-next_state transition.

        Args:
            state (State): The current state.
            action (Action): The action taken.
            next_state (State): The resulting state.

        Returns:
            float: The reward for the transition.
        """
        pass


class MarkovDecisionProcess:
    """
    Represents and solves a Markov Decision Process (MDP).

    This class implements various methods to solve the MDP, including dynamic
    programming methods (value iteration, policy iteration) and Monte Carlo methods.

    Attributes:
        env (Environment): The environment of the MDP.
        gamma (float): The discount factor for future rewards.
    """

    def __init__(self, env: Environment, gamma: float = 0.9):
        self.env = env
        self.gamma = gamma

    def value_iteration(
        self, epsilon: float = 1e-6, max_iterations: int = 1000
    ) -> Tuple[Dict[State, float], Policy]:
        """
        Solve the MDP using the Value Iteration algorithm.

        Args:
            epsilon (float): The convergence threshold.
            max_iterations (int): The maximum number of iterations to perform.

        Returns:
            Tuple[Dict[State, float], Policy]: A tuple containing the optimal value function and the optimal policy.

        Steps:
        1. Initialize the value function V(s) = 0 for all states s.
        2. Repeat until convergence or max_iterations:
           a. For each state s:
              - Compute Q(s,a) for all actions a.
              - Set V(s) = max_a Q(s,a)
           b. If the maximum change in V is less than epsilon, stop.
        3. Compute the optimal policy based on the converged value function.
        """
        states = self.env.get_states()
        V = {s: 0 for s in states}

        for _ in range(max_iterations):
            delta = 0
            for s in states:
                v = V[s]
                V[s] = max(
                    sum(
                        self.env.get_transition_prob(s, a, s_next)
                        * (self.env.get_reward(s, a, s_next) + self.gamma * V[s_next])
                        for s_next in states
                    )
                    for a in self.env.get_actions(s)
                )
                delta = max(delta, abs(v - V[s]))
            if delta < epsilon:
                break

        # Compute the optimal policy
        policy = self._extract_policy(V)

        return V, policy

    def policy_iteration(
        self, max_iterations: int = 1000
    ) -> Tuple[Dict[State, float], Policy]:
        """
        Solve the MDP using the Policy Iteration algorithm.

        Args:
            max_iterations (int): The maximum number of iterations to perform.

        Returns:
            Tuple[Dict[State, float], Policy]: A tuple containing the optimal value function and the optimal policy.

        Steps:
        1. Initialize a random policy π and value function V(s) = 0 for all states s.
        2. Repeat until convergence or max_iterations:
           a. Policy Evaluation: Compute V_π until convergence.
           b. Policy Improvement: For each state s:
              - Find a = argmax_a Q(s,a)
              - If a ≠ π(s), then π(s) = a
           c. If the policy is unchanged, stop.
        """
        states = self.env.get_states()
        V = {s: 0 for s in states}
        policy = Policy({s: np.random.choice(self.env.get_actions(s)) for s in states})

        for _ in range(max_iterations):
            # Policy Evaluation
            V = self._policy_evaluation(policy, V)

            # Policy Improvement
            policy_stable = True
            for s in states:
                old_action = policy.get_action(s)
                best_action = max(
                    self.env.get_actions(s),
                    key=lambda a: sum(
                        self.env.get_transition_prob(s, a, s_next)
                        * (self.env.get_reward(s, a, s_next) + self.gamma * V[s_next])
                        for s_next in states
                    ),
                )
                policy.state_action_map[s] = best_action
                if old_action != best_action:
                    policy_stable = False

            if policy_stable:
                break

        return V, policy

    def monte_carlo_first_visit(
        self, num_episodes: int = 10000
    ) -> Tuple[Dict[State, float], Policy]:
        """
        Solve the MDP using the Monte Carlo First-Visit method.

        Args:
            num_episodes (int): The number of episodes to simulate.

        Returns:
            Tuple[Dict[State, float], Policy]: A tuple containing the estimated value function and the derived policy.

        Steps:
        1. Initialize V(s) = 0 and N(s) = 0 for all states s.
        2. Repeat for num_episodes:
           a. Generate an episode using the current policy.
           b. For each state s in the episode:
              - If it's the first visit to s in this episode:
                * Increment N(s)
                * Update V(s) = V(s) + (G - V(s)) / N(s), where G is the return from s
        3. Compute the policy based on the estimated value function.
        """
        states = self.env.get_states()
        V = {s: 0 for s in states}
        N = {s: 0 for s in states}

        for _ in range(num_episodes):
            episode = self._generate_episode()
            G = 0
            for t in range(len(episode) - 1, -1, -1):
                s, a, r = episode[t]
                G = self.gamma * G + r
                if s not in [e[0] for e in episode[:t]]:  # First visit
                    N[s] += 1
                    V[s] += (G - V[s]) / N[s]

        policy = self._extract_policy(V)
        return V, policy

    def _policy_evaluation(
        self, policy: Policy, V: Dict[State, float], epsilon: float = 1e-6
    ) -> Dict[State, float]:
        """
        Evaluate a given policy by computing its value function.

        Args:
            policy (Policy): The policy to evaluate.
            V (Dict[State, float]): Initial value function.
            epsilon (float): Convergence threshold.

        Returns:
            Dict[State, float]: The computed value function for the given policy.
        """
        states = self.env.get_states()
        while True:
            delta = 0
            for s in states:
                v = V[s]
                a = policy.get_action(s)
                V[s] = sum(
                    self.env.get_transition_prob(s, a, s_next)
                    * (self.env.get_reward(s, a, s_next) + self.gamma * V[s_next])
                    for s_next in states
                )
                delta = max(delta, abs(v - V[s]))
            if delta < epsilon:
                break
        return V

    def _extract_policy(self, V: Dict[State, float]) -> Policy:
        """
        Extract the optimal policy from a value function.

        Args:
            V (Dict[State, float]): The value function.

        Returns:
            Policy: The extracted optimal policy.
        """
        policy = {}
        for s in self.env.get_states():
            policy[s] = max(
                self.env.get_actions(s),
                key=lambda a: sum(
                    self.env.get_transition_prob(s, a, s_next)
                    * (self.env.get_reward(s, a, s_next) + self.gamma * V[s_next])
                    for s_next in self.env.get_states()
                ),
            )
        return Policy(policy)

    def _generate_episode(self) -> List[Tuple[State, Action, float]]:
        """
        Generate an episode using the current policy.

        Returns:
            List[Tuple[State, Action, float]]: A list of (state, action, reward) tuples representing the episode.
        """
        # This is a placeholder implementation. In a real scenario, you would interact with the environment
        # to generate episodes based on the current policy and the environment dynamics.
        episode = []
        current_state = np.random.choice(self.env.get_states())
        while True:
            action = np.random.choice(self.env.get_actions(current_state))
            next_state = np.random.choice(
                self.env.get_states(),
                p=[
                    self.env.get_transition_prob(current_state, action, s)
                    for s in self.env.get_states()
                ],
            )
            reward = self.env.get_reward(current_state, action, next_state)
            episode.append((current_state, action, reward))
            if np.random.random() < 0.1:  # 10% chance to end the episode
                break
            current_state = next_state
        return episode
