from typing import Dict
from util.base.named_hashable import NamedHashable


class State(NamedHashable):
    """
    State of a Markov Decision Process.
    """

    ...


class Action(NamedHashable):
    """
    Action of a Markov Decision Process.
    """

    ...


StateValue = Dict[State, float]
ActionValue = Dict[Action, float]

ProbabilisticPolicy = Dict[State, Dict[Action, float]]
DeterministicPolicy = Dict[State, Action]

RewardFunction = Dict[State, Dict[Action, Dict[State, float]]]

StateActionTransitionProba = Dict[State, Dict[Action, Dict[State, float]]]
