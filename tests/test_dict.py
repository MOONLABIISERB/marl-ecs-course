import numpy as np
import typing as t

from marl.utils.rl.base.params import State, Action

T_State = t.TypeVar("T_State")
T_Action = t.TypeVar("T_Action")


class Policy[T_State: State, T_Action: Action](t.OrderedDict[T_State, T_Action]):
    _state_action_probas: t.OrderedDict[T_State, t.OrderedDict[T_Action, float]]

    def __init__(self, states: t.List[T_State], actions: t.List[T_Action]) -> None:
        super().__init__()
        self._states: t.Final[t.List[T_State]] = states
        self._actions: t.Final[t.List[T_Action]] = actions

    def _randomize(self) -> None:
        """
        Randomizes the policy.
        """
        state_action_proba_values = np.random.randn(
            len(self._states), len(self._actions)
        )
        state_action_proba_values = np.exp(state_action_proba_values)
        state_action_proba_values = state_action_proba_values.sum(axis=1, keepdims=True)
        self._state_action_probas = t.OrderedDict(
            {
                state: t.OrderedDict(
                    {
                        action: proba
                        for action, proba in zip(self._actions, action_proba_values)
                    }
                )
                for state, action_proba_values in zip(
                    self._states, state_action_proba_values
                )
            }
        )

    def __getitem__(self, state: T_State) -> T_Action:
        action_probas = self._state_action_probas[state]
        return self._actions[np.random.choice(1, p=list(action_probas.values()))]

    def __setitem__(self, state: T_State, action: T_Action) -> None:
        self._state_action_probas[state] = t.OrderedDict(
            {action_: 0.0 for action_ in self._actions}
        )
        self._state_action_probas[state][action] = 1.0

    def get_proba(self, s0: T_State, a: T_Action) -> float:
        return self._state_action_probas[s0][a]

    def set_probas(
        self,
        probas: np.ndarray,
        s0: T_State,
        actions: t.List[T_Action] | None = None,
        logits: bool = False,
    ) -> None:
        """
        Sets the probabilities of each action for a given state.
        The restrictions on the parameters are:
        - `s0` should be contained in the `self._states` list.
        - `actions` should contain all actions in the `self._actions` list,
            but could be in a different order. This parameter just allows
            more flexibility in the code.

        Values of `probas` are converted to a valid probability distribution using
        softmax, if `logits` is `True`. Else, the values should add up to `1.0`.

        Args:
            s0 (T_State): The state to set the action probabilities of.
            actions (List[T_Actions] | None): The actions, optional.
            probas (Iterable[float] | np.ndarray): The respective
                probabilities of each action.
            logits (bool): Whether the probabilities passed are logits.
        """
        # Validation
        try:
            assert s0 in self._states
        except AssertionError:
            raise KeyError(f"Key {s0} is not in states for the policy")
        try:
            if actions is not None:
                assert set(actions) == set(self._actions) and len(actions) == len(
                    self._actions
                )
        except AssertionError:
            raise Exception("Parameter `actions` does not match actions for the policy")
        try:
            assert np.sum(probas) - 1.0 < 1e-8
        except AssertionError:
            raise ValueError("Values of parameter `probas` should add up to 1.0")
        try:
            assert len(probas) == len(self._actions)
        except AssertionError:
            raise ValueError(
                f"Size of `probas` ({len(probas)}) and actions for the policy ({len(self._actions)}) do not match."
            )

        # Softmax on probas if logits
        if logits:
            probas = np.exp(probas)
            probas /= np.sum(probas)

        action_probas = {a: p for a, p in zip(actions or self._actions, probas)}
        self._state_action_probas[s0] = t.OrderedDict(
            {a: action_probas[a] for a in self._actions}
        )


policy = Policy[State, Action](states=[], actions=[])
