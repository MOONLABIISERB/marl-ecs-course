import pygame
import numpy as np

# Define the Multi-Agent Environment
class MultiAgentEnv():
    def __init__(self, grid_size=(6, 6), agents=None, goals=None, obstacles=None, num_agents=3, cell_size=50):
        """
        Initialize the MAPF environment.
        :param grid_size: Tuple (rows, cols) representing the size of the grid.
        :param num_agents: Number of agents in the environment.
        :param agents: Optional dictionary of agent positions {id: (row, col)}.
        :param goals: Optional dictionary of goal positions {id: (row, col)}.
        :param obstacles: List of obstacle positions as (row, col) tuples.
        """

        

        self.grid_rows = int(grid_size[0])
        self.grid_columns = int(grid_size[1])
        self.cell_size = cell_size
        
        self.obstacles = set(obstacles) if obstacles is not None else set()
        self.init_None = True if agents is None else False
        self.agents_num = num_agents
        self.goals = goals
        self.agents_init_pos = agents
        self.agents = {}
        self._initialize_agents_and_goals()
        self.rewards = {}
        self.steps = 0
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (grid_size[1] * cell_size, grid_size[0] * cell_size)
        )

        # Initialize font (this assumes you have initialized Pygame already)
        self.font = pygame.font.SysFont(None, 24)  # You can adjust the size as needed
        pygame.display.set_caption("Multi-Agent Path Finding")
        self.clock = pygame.time.Clock()

        # Colors
        self.colors = {
            "background": (240, 240, 240),
            "grid": (200, 200, 200),
            "obstacle": (100, 100, 100),
            "agent": [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)],
            "goal": [(0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0)],
        }
        
        # Define independent action spaces for each agent
        # 5 actions stay, up, down, left, right
        self.action_spaces = { i : [j for j in range(5)] for i in range(num_agents)}  # List of actions for each agent (0: up, 1: down, 2: left, 3: right, 4: stay)
        
        self.done = {}
    
    def _initialize_agents_and_goals(self):
        """
        Initialize agents and their goal positions while avoiding obstacles.
        """
        available_positions = [
            (r, c)
            for r in range(self.grid_rows)
            for c in range(self.grid_columns)
            if (r, c) not in self.obstacles
        ]
        np.random.shuffle(available_positions)

        # Use provided agents/goals or generate them
        if self.init_None:
            self.agents_init_pos = {i: available_positions.pop() for i in range(self.agents_num)}
            self.agents = self.agents_init_pos.copy()
        else:
            self.agents = self.agents_init_pos.copy()

        if self.goals is None:
            self.goals = {i: available_positions.pop() for i in range(self.agents_num)}

    def _get_observations(self):
        state = {}
        for agent_id in self.agents.keys():
            state[agent_id] = [self.agents[agent_id], self.goals[agent_id]]


        return state

    def reset(self):
        # Reset the state for each agent
        self._initialize_agents_and_goals()
        self.done = {i: False for i in range(self.agents_num)}
        self.rewards = {i: 0 for i in range(self.agents_num)}
        self.steps = 0
        # Return the initial observations for all agents
        return self._get_observations()
    
    

    def step(self, action_dict):
        """
        Move agents based on the provided actions.
        :param actions: List of actions for each agent (0: up, 1: down, 2: left, 3: right, 4: stay)
        """
        self.steps += 1
        for agent_id, action in action_dict.items():
            current_pos = self.agents[agent_id]
            next_pos = self._compute_next_position(current_pos, action)

            # Check if the move is valid
            if self._is_valid_position(next_pos):
                self.agents[agent_id] = next_pos

            self.done[agent_id] = self._is_goal_reached(agent_id)

            terminated = all(self.done.values())
        
        for agent_id in self.agents.keys():
            if self.done[agent_id]:
                self.rewards[agent_id] = 10
            else:
                self.rewards[agent_id] = -0.1
            if terminated:
                self.rewards[agent_id] = 1000

        # Update states
        observations = self._get_observations()
        
        # Return observations, rewards, done flag, and info
        return observations, self.rewards, terminated, self.steps
    
    def _compute_next_position(self, position, action):
        """
        Compute the next position based on the action.
        """
        row, col = position

        if action == 0:  # Stay
            return row, col
        elif action == 1:  # Up
            return row - 1, col
        elif action == 2:  # Down
            return row + 1, col
        elif action == 3:  # Left
            return row, col - 1
        elif action == 4:  # Righjt
            return row, col + 1

    def _is_valid_position(self, position):
        """
        Check if a position is valid (within bounds and not an obstacle).
        """
        row, col = position
        if not (0 <= row < self.grid_rows and 0 <= col < self.grid_columns):
            return False
        if position in self.obstacles or position in self.agents.values():
            return False
        return True
    
    def _is_goal_reached(self, agent_id):
        """
        Check if an agent has reached its goal.
        """
        agent_x, agent_y = self.agents[agent_id]
        goal_x, goal_y = self.goals[agent_id]

        if agent_x == goal_x and agent_y == goal_y:
            return True

        return False

    def render(self):
        """
        Render the grid world using Pygame.
        """
        # Clear screen
        self.screen.fill(self.colors["background"])

        # Draw grid
        for row in range(self.grid_rows):
            for col in range(self.grid_columns):
                rect = pygame.Rect(
                    col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size
                )
                pygame.draw.rect(self.screen, self.colors["grid"], rect, 1)

                # Draw obstacles
                if (row, col) in self.obstacles:
                    pygame.draw.rect(self.screen, self.colors["obstacle"], rect)

        # Draw goals as squares
        for goal_id, (r, c) in self.goals.items():
            goal_rect = pygame.Rect(
                    c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size
                )
            pygame.draw.rect(
                self.screen,
                self.colors["goal"][goal_id % len(self.colors["goal"])],
                goal_rect,
            )

        # Draw agents as circles
        for agent_id, (r, c) in self.agents.items():
            pygame.draw.circle(
                self.screen,
                self.colors["agent"][agent_id % len(self.colors["agent"])],
                (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2),
                self.cell_size // 3,
            )

        pygame.display.update()