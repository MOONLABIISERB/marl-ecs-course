from marl.utils.rl.base import NamedHashable
import typing as t


class State(NamedHashable):
    """A state of an agent in the environment."""

    _terminal: bool

    def __init__(self, name: str | None = None, terminal: bool = False) -> None:
        super().__init__(name or "STATE")
        self._terminal: t.Final[bool] = terminal

    @property
    def is_terminal(self) -> bool:
        return self._terminal


class Action(NamedHashable):
    """An action that can be performed by the agent."""

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name or "ACTION")

