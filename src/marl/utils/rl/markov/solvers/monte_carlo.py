import numpy as np
from marl.utils.rl.base.params import State, Action
from marl.utils.rl.base.transition import TransitionProbaManager
from marl.utils.rl.base.reward import RewardFuncManager
from marl.utils.rl.base.policy import Policy
from marl.utils.rl.markov.process import MarkovDecisionProcess
from marl.utils.rl.markov.solvers import MarkovDecisionProcessSolver
import typing as t
from typing_extensions import override

T_State = t.TypeVar("T_State")
T_Action = t.TypeVar("T_Action")


class MonteCarloSolver[T_State: State, T_Action: Action](
    MarkovDecisionProcessSolver[T_State, T_Action]
):
    def __init__(self) -> None:
        super().__init__(name="MonteCarloSolver")

    def calculate_episode_returns(
        self,
        episode: t.List[t.Tuple[T_State, T_Action, float]],
    ) -> t.List[float]:
        g_episode = []
        # TODO Calculate returns of the episode here
        return g_episode

    @override
    def solve(
        self, mdp: MarkovDecisionProcess[T_State, T_Action]
    ) -> t.Tuple[t.OrderedDict[T_State, float], Policy[T_State, T_Action]]:
        pass
