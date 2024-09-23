import typing as t

from marl.utils.rl.base.params import State, Action
from marl.utils.rl.base.transition import TransitionProbaManager
from marl.utils.rl.base.reward import RewardFuncManager
import numpy as np


T_State = t.TypeVar("T_State")
T_Action = t.TypeVar("T_Action")


class Environment[T_State: State, T_Action: Action]:
    """
    The environment for a reinforcement learning setup.

    Attributes:
        _states (List[T_State]): The list of states.
        _actions (List[T_Action]): The list of actions.
        _transition_proba_manager (TransitionProbaManager):
            The transition probability manager for the environment.
        _reward_func_manager (RewardFunctionManager):
            The reward function manager for the environment.
    """

    _states: t.List[T_State]
    _actions: t.List[T_Action]
    _transition_proba_manager: TransitionProbaManager
    _reward_func_manager: RewardFuncManager

    def __init__(self, states: t.List[T_State], actions: t.List[T_Action]) -> None:
        self._states = states
        self.actions = actions

    def interact(self, state: T_State, action: T_Action) -> t.Tuple[T_State, float]:
        """
        The agent interacts with the environment by performing some `action`
        at some `state`.

        Args:
            state (T_State): The start state of the agent.
            action (T_Action): The action performed by the agent

        Returns:
            Tuple[T_State, float]: A tuple containing a final state and the
                reward obtained by the agent by going from current state to
                final state by performing action.
        """
        final_state: T_State = np.random.choice(
            np.arange(len(self._states)),
            p=[
                self._transition_proba_manager.get_proba(a=action, s0=state, s=s)
                for s in self._states
            ],
        )
        reward: float = self._reward_func_manager.get(s0=state, a=action, s=final_state)
        return final_state, reward
