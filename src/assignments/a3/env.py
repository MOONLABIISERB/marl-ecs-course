import torch
import numpy as np
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass
from gymnasium.core import Env
from typing_extensions import override
from numpy import typing as npt

GridCoord = Tuple[int, int]


@dataclass
class Action:
    NOPE = (0, 0)
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)


class Agent:
    _coords: npt.NDArray[np.int16]

    def __init__(self, coords: npt.NDArray[np.int16]):
        self._coords = coords


class GridWorldEnv(Env):
    _state: torch.Tensor
    _agents_start: List[GridCoord]

    def __init__(
        self,
        width: int,
        height: int,
        n_agents: int,
        agents_start: List[GridCoord],
        walls: List[GridCoord],
        goals: List[GridCoord],
    ) -> None:
        self._agents_start = agents_start
        self._state = torch.zeros(3, height, width, dtype=torch.bool)
        self._state[0, agents_start] = True
        self._state[1, walls] = True
        self._state[2, goals] = True
