from marl.utils.rl.base.params import State, Action
from marl.utils.rl.markov.process import MarkovDecisionProcess
from marl.utils.rl.base.policy import Policy
import typing as t
from abc import ABC, abstractmethod
import random


class Episode[
    T_State: State,
    T_Action: Action,
](t.List[t.Tuple[T_State, T_Action, float]], ABC):
    _mdp: MarkovDecisionProcess[T_State, T_Action]
    _current_state: T_State

    def __init__(self, mdp: MarkovDecisionProcess[T_State, T_Action]) -> None:
        self._mdp = mdp
        self._current_state = self._mdp.states[0]
        self.reset()

    def reset(self) -> None:
        self.clear()

    @abstractmethod
    def _step(self, action: T_Action) -> t.Tuple[T_State, float]:
        pass

    @abstractmethod
    def _should_terminate(self) -> bool:
        """
        Determines whether this episode should terminate.

        Returns:
            bool: Whether this episode should terminate.
        """
        pass

    def generate(self, max_len: int, policy: Policy[T_State, T_Action]) -> None:
        self.reset()
        # Choose start state
        self._current_state: T_State = random.choice(seq=self._mdp.states)
        for i in range(max_len):
            self._current_state = self[-1][0]
            next_state: T_State
            action: T_Action = policy[self._current_state]
            next_state, reward = self._step(action=action)
            self.append((self._current_state, action, reward))
            self._current_state = next_state
