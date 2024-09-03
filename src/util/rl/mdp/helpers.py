from typing import List
import numpy as np
from util.rl.mdp.params import (
    Action,
    ProbabilisticPolicy,
    DeterministicPolicy,
    State,
    StateActionTransitionProba,
)


def create_deterministic_policy(policy: ProbabilisticPolicy) -> DeterministicPolicy:
    """
    Create deterministic policy from a probabilistic policy by considering
    the action with the highest probability for each state.

    Args:
        policy (ProbabilisticPolicy): The probabilistic policy.

    Returns:
        DeterministicPolicy: The determinstic policy created by maximising
            the probability over actions for each of the states.
    """
    policy_: DeterministicPolicy = {state: None for state in policy}
    for state in policy:
        actions = policy[state]
        max_action: Action = list(policy[state].keys())[0]
        for action in actions:
            if policy[state][action] > policy[state][max_action]:
                max_action = action
        policy_[state] = max_action
    return policy_


def proba_policy_to_det(proba_policy: ProbabilisticPolicy) -> DeterministicPolicy:
    return {
        state: list(proba_policy[state].keys())[np.argmax(proba_policy[state].values())]
        for state in proba_policy
    }


def det_policy_to_proba(
    det_policy: DeterministicPolicy, actions: List[Action]
) -> ProbabilisticPolicy:
    return {
        state: {
            action: 1.0 if action == det_policy[state] else 0.0 for action in actions
        }
        for state in det_policy
    }


def create_action_transition_proba_matrix(
    transition_proba: StateActionTransitionProba, action: Action
) -> np.ndarray:
    """
    Creates a state transition probability matrix from a
    StateActionTransitionProba object.

    Args:
        transition_proba (StateActionTransitionProba): The object to create
            the transition probability matrix from.
        action (Action): The action for which we need this transition
            probability matrix.

    Returns:
        np.ndarray: The transition probability matrix as a numpy array with
            shape (m, m) where m is the number of states in transition_proba.
    """
    states: List[State] = [state for state in transition_proba]
    transition_proba_matrix: np.ndarray = np.array(
        [[transition_proba[s1][action][s2] for s2 in states] for s1 in states]
    )

    return transition_proba_matrix
