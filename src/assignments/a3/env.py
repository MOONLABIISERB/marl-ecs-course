import numpy as np
from enum import Enum
from numpy import typing as npt
from typing import Dict, List, Tuple, Set


class Action(Enum):
    NOPE = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


GridPoint = Tuple[int, int]


movement: Dict[Action, GridPoint] = {
    Action.NOPE: (0, 0),
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}


class Agent:
    _coords: GridPoint

    def __init__(self, coords: GridPoint):
        self._coords = coords

    def coords_on_action(self, action: Action) -> GridPoint:
        x, y = self.coords
        dx, dy = movement[action]
        return (x + dx, y + dy)

    @property
    def coords(self) -> GridPoint:
        return self._coords

    @coords.setter
    def coords(self, _coords: GridPoint) -> None:
        self._coords = _coords


class GridWorldEnv:
    _agents_start: List[GridPoint]
    _walls: Set[GridPoint]
    _goals: List[GridPoint]
    _agents: List[Agent]
    _grid_size: Tuple[int, int]
    _agent_fov: Tuple[int, int]
    _steps: int
    _max_steps: int
    _penalty: float

    def __init__(
        self,
        width: int,
        height: int,
        agents_start: List[GridPoint],
        walls: Set[GridPoint],
        goals: List[GridPoint],
        agent_fov: Tuple[int, int],
        max_steps: int,
    ) -> None:
        self._grid_size = height, width
        self._agents = [Agent(coords=coords) for coords in agents_start]
        self._walls = walls
        self._goals = goals
        self._agent_fov = agent_fov
        self._steps = 0
        self._max_steps = max_steps
        self._agents_start = agents_start
        self._penalty = 0.0

    @property
    def n_agents(self) -> int:
        return len(self._agents)

    @property
    def agent_coords(self) -> List[GridPoint]:
        return [agent.coords for agent in self._agents]

    def _is_valid_grid(self, coords: GridPoint) -> bool:
        y, x = coords
        height, width = self._grid_size
        return 0 <= x < width and 0 <= y < height

    def reset(self) -> npt.NDArray[np.int8]:
        self._steps = 0
        self._penalty = 0.0
        for agent, coords in zip(self._agents, self._agents_start):
            agent.coords = coords
        return self.get_observation()

    def get_observation_for_agent(self, index: int) -> npt.NDArray[np.int8]:
        h, w = self._agent_fov
        obs_matrix = np.zeros(shape=(3, 2 * h + 1, 2 * w + 1), dtype=np.int8)
        agent_y, agent_x = self._agents[index].coords
        for i, obs_grid_y in enumerate(range(-h, h + 1)):
            for j, obs_grid_x in enumerate(range(-w, w + 1)):
                grid_coords = obs_grid_y + agent_y, obs_grid_x + agent_x
                if not self._is_valid_grid(coords=grid_coords):
                    obs_matrix[:, i, j] = -1
                    self._penalty += 10
                else:
                    obs_matrix[0, i, j] = grid_coords in self.agent_coords
                    obs_matrix[1, i, j] = grid_coords in self._walls
                    obs_matrix[2, i, j] = grid_coords in self._goals
        return obs_matrix

    def get_observation(self) -> npt.NDArray[np.int8]:
        return np.array(
            [self.get_observation_for_agent(index=i) for i in range(len(self._agents))]
        )

    def step(
        self, actions: List[Action]
    ) -> Tuple[npt.NDArray[np.int8], float, List[bool], bool, bool]:
        assert len(actions) == len(
            self._agents
        ), "Number of actions should be same as number of agents"

        self._penalty: float = 0.0
        agents_at_goal: List[bool] = [False] * len(self._agents)
        done = False

        for i, (agent, action, agent_goal) in enumerate(
            zip(self._agents, actions, self._goals)
        ):
            coords_on_action = agent.coords_on_action(action=action)
            if (
                action is not Action.NOPE
                and self._is_valid_grid(coords=coords_on_action)
                and not (
                    (coords_on_action in self.agent_coords)
                    or (coords_on_action in self._walls)
                )
            ):
                agent.coords = coords_on_action
            if agent.coords == agent_goal:
                self._penalty -= 1.0
                agents_at_goal[i] = True
            else:
                self._penalty += 1.0

            if all(agents_at_goal):
                self._penalty -= 500.0
                done = True

        self._steps += 1
        terminated = self._steps >= self._max_steps

        return self.get_observation(), self._penalty, agents_at_goal, done, terminated


def main():
    walls = {
        (0, 4),
        (1, 4),
        (2, 4),
        (2, 5),
    }
    agents = [
        (1, 1),
    ]

    goals = [
        (5, 8),
    ]

    env = GridWorldEnv(
        width=10,
        height=10,
        agents_start=agents,
        walls=walls,
        goals=goals,
        agent_fov=(2, 2),
        max_steps=3,
    )

    print(env.get_observation_for_agent(index=0))
    print(env.step(actions=[Action.RIGHT]))
    print(env.get_observation_for_agent(index=0))
    print(env.step(actions=[Action.RIGHT]))
    print(env.get_observation().flatten())
    print(env.step(actions=[Action.RIGHT]))
    print(env.get_observation().flatten())


if __name__ == "__main__":
    main()
