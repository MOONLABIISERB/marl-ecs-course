import numpy as np
from marl.utils.rl.base.params import State, Action
from marl.utils.rl.base.policy import Policy
from marl.utils.rl.base.reward import RewardFuncManager
from marl.utils.rl.base.transition import TransitionProbaManager
from marl.utils.rl.markov.process import MarkovDecisionProcess
from marl.utils.rl.markov.solvers.value_iter import ValueIterationSolver
from marl.utils.rl.markov.solvers.policy_iter import PolicyIterationSolver
import typing as t
from typing_extensions import override


Coords = np.ndarray


class SokobanState(State):
    """
    The state of a sokoban game.
    """

    _agent_coords: Coords
    _box_coords: Coords

    def __init__(self, agent_coords: Coords, box_coords: Coords) -> None:
        super().__init__(name="SOKOBAN_STATE")
        self.agent_coords = agent_coords
        self.box_coords = box_coords

    @override
    def _repr(self) -> str:
        return f"agent={self._agent_coords}; box={self.box_coords}"

    def __eq__(self, value) -> bool:
        return (
            self.agent_coords == object.agent_coords
            and self.box_coords == object.box_coords
        )


class SokobanAction(Action):
    """
    The possible actions in a Sokoban game.
    """

    _x: int
    _y: int
    _action_name: t.Literal["LEFT", "RIGHT", "UP", "DOWN"]

    def __init__(
        self, x: int, y: int, action_name: t.Literal["LEFT", "RIGHT", "UP", "DOWN"]
    ) -> None:
        super().__init__(name="SOKOBAN_ACTION")
        self._x = x
        self._y = y
        self._action_name = action_name


class SokobanWorld:
    _states: t.List[SokobanState]

    def __init__(self, height: int, width: int) -> None:
        self._states = [
            SokobanState(agent_coords=(agent_x, agent_y), box_coords=(box_x, box_y))
            for agent_x in range(width)
            for agent_y in range(height)
            for box_x in range(width)
            for box_y in range(height)
            if (agent_x, agent_y) != (box_x, box_y)
        ]

    def get_state(self, agent_coords: Coords, box_coords: Coords) -> SokobanState:
        """
        Gets a SokobanState based on the coordinates of the agent and the box.
        """
        pass
