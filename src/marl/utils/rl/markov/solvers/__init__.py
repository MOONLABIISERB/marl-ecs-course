from abc import ABC, abstractmethod
import typing as t
from marl.utils.rl.markov.process import MarkovDecisionProcess
from marl.utils.rl.base.params import State, Action
from marl.utils.rl.base.policy import Policy

T_State = t.TypeVar("T_State")
T_Action = t.TypeVar("T_Action")


class MarkovDecisionProcessSolver[T_State: State, T_Action: Action](ABC):
    name: str

    def __init__(self, name: str = "Solver") -> None:
        self.name = name

    @abstractmethod
    def solve(
        self,
        mdp: MarkovDecisionProcess[T_State, T_Action],
    ) -> t.Tuple[t.OrderedDict[T_State, float], Policy[T_State, T_Action]]:
        """
        Solves a Markov Decision Process.

        Args:
            mdp (MarkovDecisionProcess): The Markov Decision Process.

        Returns:
            Tuple[OrderedDict[T_State, float], Policy]:
                A tuple containing:
                - Optimal value function as an OrderedDict[T_State, float]
                - Optimal policy
        """
        pass
