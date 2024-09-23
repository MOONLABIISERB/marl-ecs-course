import typing as t

import numpy as np

from marl.utils.rl.base.params import Action, State


T_State = t.TypeVar("T_State")
T_Action = t.TypeVar("T_Action")

TransitionProba = t.OrderedDict[
    T_Action, t.OrderedDict[T_State, t.OrderedDict[T_State, float]]
]


class TransitionProbaManager[T_State: State, T_Action: Action]:
    """
    Manages the transition probabilities for the states, actions and final
    states in an environment.

    Attributes:
        _transition_proba: TransitionProba
    """

    _transition_proba: TransitionProba

    def __init__(self, states: t.List[T_State], actions: t.List[T_Action]) -> None:
        """
        Initializes a TransitionProbaManager object.

        Args:
            states (List[T_State]): The list of states.
            actions (List[T_Action]): The list of actions.
        """
        # a = Action; s0 = Start state; s = Final state
        self._transition_proba = t.OrderedDict(
            {
                a: t.OrderedDict(
                    {s0: t.OrderedDict.fromkeys(states, 0.0) for s0 in states}
                )
                for a in actions
            }
        )

    def get_proba(self, a: T_Action, s0: T_State, s: T_State) -> float:
        """
        Gets the value of p(s | s0, a).

        Args:
            a (Action): The action.
            s0 (State): The start state.
            s (State): The final state.

        Returns:
            float: The value of p(s | s0, a)
        """
        return self._transition_proba[a][s0][s]

    def set_proba(self, a: T_Action, s0: T_State, s: T_State, proba: float) -> None:
        """
        Sets the value of p(s | s0, a).

        Args:
            a (T_Action): The action.
            s0 (T_State): The start state.
            s (T_State): The final state.
        """
        self._transition_proba[a][s0][s] = proba

    def get_action_transition_proba_matrix(self, action: T_Action) -> np.ndarray:
        """
        Gets the transition probability matrix for the given action.

        Args:
            action (T_Action): The action to get the transition probability matrix for.

        Returns:
            np.ndarray: The transition probability matrix.
        """
        return np.array(
            [
                [
                    self._transition_proba[action][s0][s]
                    for s in self._transition_proba[action][s0]
                ]
                for s0 in self._transition_proba[action]
            ]
        )
