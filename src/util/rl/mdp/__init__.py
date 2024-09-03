from copy import deepcopy
from typing import List, Tuple
import numpy as np
from util.rl.mdp.helpers import create_deterministic_policy, det_policy_to_proba
from util.rl.mdp.params import (
    State,
    Action,
    StateValue,
    DeterministicPolicy,
    ProbabilisticPolicy,
    RewardFunction,
    StateActionTransitionProba,
)


class MarkovDecisionProcess:
    """
    Class for a Markov Decision Process.

    Attributes:
        _states (List[State]): The list of states.
        _actions (List[Action]): The list of actions.
        _transition_probas (StateTranstionProba): A dict with actions
            as keys and a state transition probability as the keys.
            For example, to get p(s2 | s1, a) use:
            >>> self._transition_probas[s1][a][s2]
        _reward (RewardFunction): The reward function.
            For example, to get the reward when the agent goes from state s1
            to state s2 by performing action a, use:
            >>> self._reward[s1][a][s2]
        _values (StateValue): The state value function.
    """

    _states: List[State]
    _actions: List[Action]
    _transition_probas: StateActionTransitionProba
    _reward: RewardFunction
    _values: StateValue

    def __init__(
        self,
        name: str,
        states: List[State],
        actions: List[Action],
        transition_probas: StateActionTransitionProba,
        reward: RewardFunction,
    ) -> None:
        self.name = name
        self._states = states
        self._actions = actions
        self._transition_probas = transition_probas
        self._reward = reward
        self._values = {state: 0.0 for state in states}

    def _randomize_state_values(self) -> None:
        """
        Randomizes the state value function for this MDP.
        """
        for state in self._values:
            self._values[state] = np.random.randn()

    def _create_random_proba_policy(self) -> ProbabilisticPolicy:
        """
        Creates a random probabilistic policy.
        """
        policy: ProbabilisticPolicy = {
            state: {action: 0.0 for action in self._actions} for state in self._states
        }
        for state in self._states:
            probas = np.random.randn(len(self._actions))
            probas = np.exp(probas)
            probas /= probas.sum()
            for action, proba in zip(self._actions, probas):
                policy[state][action] = proba
        return policy

    def _create_random_det_policy(self) -> DeterministicPolicy:
        """
        Creates a random deterministic policy.
        """
        return create_deterministic_policy(policy=self._create_random_proba_policy())

    # def _get_policy_transition_proba_matrix()

    def _set_forbidden_state(self, forbidden_state: State) -> None:
        """
        Makes a state forbidden. Useful for creating walls or obstacles
        in pathfinding problems, for example. A state is forbidden when:

        1. The agent cannot transit to this state from any other state.
        2. The agent, if starts from this state, cannot transit to any other
            state except this state.

        Args:
            state (State): The state to make forbidden.
        """
        for action in self._actions:
            for state in self._states:
                self._transition_probas[forbidden_state][action][state] = float(
                    state == forbidden_state
                )
                self._transition_probas[state][action][forbidden_state] = 0.0
                self._reward[state][action] = {s: -np.inf for s in self._states}

    def _evaluate_policy(
        self,
        policy: ProbabilisticPolicy | DeterministicPolicy,
        gamma: float,
        theta: float,
    ) -> StateValue:
        """
        Sets the state values according to the policy provided.

        Args:
            policy (ProbabilisticPolicy | DeterministicPolicy): The policy.
            gamma (float): The discount factor.
            theta (float): Tolerance parameter, the loop stops if the maximum
                change in value of any state goes below this parameter for
                any iteration.

        Returns:
            StateValue: The state value function according to the policy.
        """
        delta: float = np.inf
        v: StateValue = {**self._values}

        if isinstance(policy[self._states[0]], Action):
            policy = det_policy_to_proba(det_policy=policy, actions=self._actions)

        while not delta < theta:
            v_old = {**v}
            print(f"Old state value function:\n{v_old}")
            for state in self._states:
                v[state] = np.sum(
                    [
                        policy[state][action]
                        * np.sum(
                            [
                                (
                                    self._reward[state][action][next_state]
                                    # print(self._reward[state][action])
                                    + gamma
                                    * self._transition_probas[state][action][next_state]
                                    * v_old[next_state]
                                )
                                for next_state in self._states
                            ]
                        )
                        for action in self._actions
                    ]
                )
            delta = np.max(
                [
                    delta,
                    *[
                        abs(value_curr - value_old)
                        for value_curr, value_old in zip(v.values(), v_old.values())
                    ],
                ]
            )
        return v

    def _create_best_det_policy(self, gamma: float) -> DeterministicPolicy:
        """
        Create a policy from current value function which maximises the
        returns from the current state.

        Returns:
            DeterministicPolicy: The deterministic policy maximizing the
                returns from current state.
        """
        policy: DeterministicPolicy = {state: None for state in self._states}
        for state in self._states:
            best_action: Action = self._actions[0]
            g_max: float = -np.inf
            for action in self._actions:
                g = np.sum(
                    [
                        self._reward[state][action][next_state]
                        + gamma
                        * self._transition_probas[state][action][next_state]
                        * self._values[next_state]
                        for next_state in self._states
                    ]
                )
                if g > g_max:
                    best_action = action
                    g_max = g
            policy[state] = best_action
        return policy

    def solve_policy_iteration(
        self, gamma: float = 0.5
    ) -> Tuple[StateValue, DeterministicPolicy]:
        """
        Solves the MDP using **Policy Iteration**.

        Args:
            gamma (float): The discount factor.

        Returns:
            Tuple[StateValue, DeterministicPolicy]: The tuple of state value
                function and the determinstic policy obtained by solving the
                MDP.
        """
        self._randomize_state_values()
        policy = self._create_random_det_policy()

        policy_stable: bool = False

        for i in range(10):
            old_policy: DeterministicPolicy = {**policy}
            self._values = self._evaluate_policy(policy=policy, gamma=gamma, theta=1e4)
            policy = self._create_best_det_policy(gamma=gamma)
            print(policy)
            # policy_stable = old_policy == policy

        return self._values, policy

