import typing as t
from marl.utils.rl.base.params import State, Action


T_State = t.TypeVar("T_State")
T_Action = t.TypeVar("T_Action")


RewardFunction = t.Dict[T_State, t.Dict[T_Action, t.Dict[T_State, float]]]


class RewardFuncManager[T_State: State, T_Action: Action]:
    """
    The reward function for an environment.

    Attributes:
        _reward_func (RewardFunction):
            The quantity r(s0, a, s) is stored as `rewards[s0][a][s]`.
    """

    _reward_func: RewardFunction

    def __init__(self, states: t.List[T_State], actions: t.List[T_Action]) -> None:
        """
        Initializes a RewardFunction object.

        Args:
            states (List[T_State]): The list of states.
            actions (List[T_Action]): The list of actions.
        """
        # Initializes all rewards to zero
        self._reward_func = {
            state: {action: {state: 0.0 for state in states} for action in actions}
            for state in states
        }

    @property
    def reward(self) -> RewardFunction:
        """
        Returns a copy of the raw reward function.

        Returns:
            RewardFunction: The Dict representation of the reward function.
        """
        return self._reward_func

    def set(
        self,
        reward: float,
        s0: T_State,
        a: T_Action | None = None,
        s: T_State | None = None,
    ) -> None:
        """
        Sets the value for r(s0, a, s).
        - If no value provided for `a`, sets reward for all actions and all
            states from `s0` to `reward`.
        - If no value provided for `s`, sets reward for all states from `s0`
            on taking action `a`, to `reward`.

        Args:
            s0 (T_State): The current state. [required]
            a (T_Action): The action to be taken at state `s0`. [default=None]
            s (T_State): The final state reached by taking action `a` at `s0`.
                [default=None]
        """
        if a is not None:
            if s is not None:
                self._reward_func[s0][a][s] = reward
            else:
                for s in self._reward_func[s0][a]:
                    self.set(s0=s0, a=a, s=s, reward=reward)
        else:
            for action in self._reward_func[s0]:
                self.set(s0=s0, a=action, reward=reward)

    def get(self, s0: T_State, a: T_Action, s: T_State) -> float:
        """
        Gets the value of r(s0 | a, s).

        Args:
            s0 (T_State): The start state.
            a (T_Action): The action to be taken at state `s0`,
            s (T_State): The final state reached by taking action `a` at `s0`.
        """
        return self._reward_func[s0][a][s]
