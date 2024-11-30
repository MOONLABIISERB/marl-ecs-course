from typing import List, Tuple, Dict, Set
import heapq
from dataclasses import dataclass, field
from typing import Any
from env import Action


@dataclass(order=True)
class PrioritizedNode:
    priority: float
    item: Any = field(compare=False)


class AStarPlanner:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, pos: Tuple[int, int], obstacles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        neighbors = []
        for dx, dy in self.directions:
            new_pos = (pos[0] + dx, pos[1] + dy)

            # Check bounds
            if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
                continue

            # Check obstacles
            if new_pos in obstacles:
                continue

            neighbors.append(new_pos)
        return neighbors

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: List[Tuple[int, int]],
        other_agents: Dict[int, Tuple[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Find path using A* algorithm.

        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            obstacles: List of obstacle positions
            other_agents: Dictionary of other agent positions to avoid (optional)

        Returns:
            List of positions forming the path, or empty list if no path found
        """
        if start == goal:
            return [start]

        # Convert obstacles to set for O(1) lookup
        obstacle_set = set(obstacles)
        if other_agents:
            # Add other agents' positions as temporary obstacles
            obstacle_set.update(other_agents.values())

        # Priority queue for open nodes
        open_set = []
        heapq.heappush(open_set, PrioritizedNode(0, (start, 0)))  # (pos, g_score)

        # Tracking dictionaries
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}

        while open_set:
            current = heapq.heappop(open_set).item[0]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current, obstacle_set):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, PrioritizedNode(f_score[neighbor], (neighbor, g_score[neighbor])))

        return []  # No path found

    def get_next_action(self, current_pos: Tuple[int, int], next_pos: Tuple[int, int]) -> Action:
        """Convert position change to action."""
        dx = next_pos[1] - current_pos[1]  # Note: positions are (row, col)
        dy = next_pos[0] - current_pos[0]

        if dx == 1:
            return Action.RIGHT
        elif dx == -1:
            return Action.LEFT
        elif dy == 1:
            return Action.DOWN
        elif dy == -1:
            return Action.UP
        return Action.STAY

    def plan_action(self, agent_id: int, env) -> Action:
        """Plan next action for an agent using A* pathfinding."""
        if agent_id in env.agents_done:
            return Action.STAY

        current_pos = env.agents_pos[agent_id]
        goal_pos = env.goals[agent_id]

        # Get other agents' positions
        other_agents = {other_id: pos for other_id, pos in env.agents_pos.items() if other_id != agent_id}

        # Find path considering obstacles and other agents
        path = self.find_path(current_pos, goal_pos, env.obstacles, other_agents)

        if not path or len(path) < 2:
            # No path found or already at goal
            return Action.STAY

        # Get action to reach next position in path
        next_pos = path[1]  # Second position in path (first is current)
        return self.get_next_action(current_pos, next_pos)
