import typing as t
from marl.utils.rl.base.params import State, Action, T_State, T_Action
from marl.utils.rl.base.policy import Policy
from marl.utils.rl.base.reward import RewardFuncManager
from marl.utils.rl.base.transition import TransitionProba
from marl.utils.rl.markov.process import MarkovDecisionProcess
from marl.utils.rl.markov.solvers import MarkovDecisionProcessSolver as MDPSolver
from dataclasses import dataclass
import numpy as np
from numpy import typing as npt
from typing_extensions import override


class TSPState(State):
    _coords: npt.NDArray[np.float32]

    def __init__(self, coords: npt.NDArray[np.float32], terminal: bool = False) -> None:
        super().__init__(name="TSP_STATE", terminal=terminal)
        self._coords = coords

    @override
    def _repr(self) -> str:
        return f"coords={self._coords.tolist()}"


class TSPAction(Action):
    _to_state: TSPState

    def __init__(self, to_state: TSPState) -> None:
        super().__init__(name="TSP_ACTION")
        self._to_state = to_state

    @override
    def _repr(self) -> str:
        return f"to={self._to_state._coords.tolist()}"


@dataclass
class TSP[T_State: State, T_Action: Action](MarkovDecisionProcess[T_State, T_Action]):
    num_states: int
    profits: npt.NDArray[np.float32]
    area_bounds: t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]

    def reset(self) -> None:
        pass
