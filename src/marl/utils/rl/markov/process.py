from dataclasses import dataclass
import typing as t

from marl.utils.rl.base.params import Action, State
from marl.utils.rl.base.reward import RewardFuncManager
from marl.utils.rl.base.transition import TransitionProbaManager


T_State = t.TypeVar("T_State")
T_Action = t.TypeVar("T_Action")


@dataclass
class ModelBased[T_State: State, T_Action: Action]:
    transition_proba: TransitionProbaManager[T_State, T_Action] | None
    rewards: RewardFuncManager[T_State, T_Action] | None


@dataclass
class MarkovDecisionProcess[T_State: State, T_Action: Action]:
    states: t.List[T_State]
    actions: t.List[T_Action]
    gamma: float


@dataclass
class ModelBasedMarkovDecisionProcess[T_State: State, T_Action: Action](
    ModelBased[T_State, T_Action], MarkovDecisionProcess[T_State, T_Action]
):
    pass
