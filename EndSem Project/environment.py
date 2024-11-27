import math
import numpy as np
import pygame

# Constants
GRID_SIZE = 30
CELL_SIZE = 30  # Each cell's width and height in pixels
SCREEN_SIZE = GRID_SIZE * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (124, 252, 0)  # Starting line color
BLUE = (0, 0, 255)     # Finish line color
GRAY = (128, 128, 128)  # Track color
CAR_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Car colors (extendable)

class Car:
    def __init__(self, start_pos, agent_id: int, teammate_id: int):
        self.agent_id = agent_id
        self.teammate_id = teammate_id
        self.position = start_pos
        self.initial_position = start_pos
        self.max_angle = 0
        self.angle = 0
        self.checkpoint_counters = 0
        self.collision_counter = 0
        self.total_distance_traveled = 0
        self.reward = 0
        self.done = False
        self.steps_since_last_checkpoint = 0

    def reset(self, position, observation):
        self.position = position
        self.initial_position = position
        self.max_angle = 0
        self.angle = 0
        self.checkpoint_counters = 0
        self.collision_counter = 0
        self.total_distance_traveled = 0
        self.reward = 0
        self.done = False
        self.steps_since_last_checkpoint = 0
        self.observation = observation

def create_track(grid_size: int, track_width: int):
    num_checkpoints = 12
    track = set()
    checkpoints = []
    # Define the center and radius for the circular track
    center_x, center_y = grid_size // 2, grid_size // 2
    outer_radius = (grid_size // 2) - 2  # Slightly smaller to ensure visibility
    inner_radius = outer_radius - track_width

    # Create the track
    for y in range(grid_size):
        for x in range(grid_size):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if inner_radius <= distance < outer_radius:
                track.add((x, y))

    # Calculate angular positions of checkpoints
    angle_step = 360 / num_checkpoints
    for i in range(num_checkpoints):
        checkpoint = []
        angle = (i + 4) * angle_step  # Evenly distributed angles
        rad_angle = math.radians(angle)
        
        # Calculate the base point on the middle of the track
        mid_radius = (inner_radius + outer_radius) / 2
        base_x = center_x + mid_radius * math.cos(rad_angle)
        base_y = center_y + mid_radius * math.sin(rad_angle)
        
        # Decide if the checkpoint is horizontal or vertical
        is_horizontal = abs(math.sin(rad_angle)) < 0.707  # cos(45°) ≈ 0.707
        
        for w in range(-track_width - 1, track_width + 2):
            if is_horizontal:
                # Horizontal checkpoint (on sides)
                x = int(base_x + w)
                y = int(base_y)
            else:
                # Vertical checkpoint (top/bottom)
                x = int(base_x)
                y = int(base_y + w)
            
            if 0 <= x < grid_size and 0 <= y < grid_size and (x, y) in track:
                checkpoint.append((x, y))
        
        if checkpoint:
            checkpoints.append(checkpoint)

    # Create the start line
    start_line = []
    start_x = center_x  # Center position
    for y in range(grid_size // 2, grid_size):  # Only lower half of the grid
        if (start_x, y) in track:
            start_line.append((start_x, y))

    # Ensure the last checkpoint overlaps the start line
    if checkpoints:
        checkpoints[-1].extend(start_line)

    return track, checkpoints, start_line

def angular_displacement(prev_angle, grid_size, pos_car):
    x_c, y_c = grid_size // 2, grid_size // 2

    x1 = pos_car[0]
    y1 = pos_car[1]

    # Compute the current angle (in radians) using atan2
    current_angle = math.atan2(y1 - y_c, x1 - x_c)
    
    # Convert to degrees for easier interpretation (optional)
    current_angle_deg = math.degrees(current_angle)

    # Adjust the angle so that the lowest point (x_c, y_c - r) corresponds to 0 degrees
    current_angle_deg -= 90

    # Floor the angles to nearest integer values
    current_angle_deg = math.floor(current_angle_deg)
    prev_angle_deg = math.floor(prev_angle)

    # Compute angular displacement in degrees
    delta_theta = current_angle_deg - prev_angle_deg

    # Normalize the displacement to make sure the angle stays within [-360, 360] degrees
    if delta_theta > 180:
        delta_theta -= 360  # If we go over 180, adjust to the negative equivalent
    elif delta_theta < -180:
        delta_theta += 360  # If we go below -180, adjust to the positive equivalent

    # Update the total angle (also floored to nearest integer)
    total_angle = prev_angle_deg + delta_theta
    total_angle = math.floor(total_angle)  # Ensure total displacement is an integer

    return total_angle

class MultiCarRacing():
    def __init__(self, n_cars: int = 4, grid_size: int = 30, track_width: int = 5, render_mode=None):
        super().__init__()
        self.n_cars = n_cars
        self.grid_rows = grid_size
        self.grid_columns = grid_size
        self.render_mode = render_mode
        
        # Create track first
        self.track, self.checkpoints, self.start_line = create_track(
            grid_size, track_width
        )
        
        # Initialize agents with starting positions
        self.agents = {}
        for agent_id in range(n_cars):
            if (agent_id % 2 == 0):
                teammate_id = agent_id + 1
            else:
                teammate_id = agent_id - 1
            start_pos = self.start_line[agent_id % len(self.start_line)]
            self.agents[agent_id] = Car(start_pos, agent_id, teammate_id)

        self.observation_space = {
            agent_id: np.concatenate([
                np.zeros((grid_size, grid_size)).flatten(),  # Flattened grid of zeros (grid_size * grid_size)
                np.array([agent.angle]),  # Current agent's angle
                np.array([self.agents[agent.teammate_id].angle]),  # Teammate's angle
                np.array([
                    other_agent.angle
                    for other_id, other_agent in self.agents.items()
                    if other_id != agent_id and other_id != agent.teammate_id  # Angles of all other agents (excluding current agent and teammate)
                ])
            ])
            for agent_id, agent in self.agents.items()
        }

        # Pygame Setup for rendering
        if render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
            pygame.display.set_caption("Multi-Agent Racetrack Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

    def reset(self):
        # Reset each agent to its starting position
        for agent_id, agent in self.agents.items():
            start_pos = self.start_line[agent_id % len(self.start_line)]
            
            agent.reset(start_pos, self.get_observation(agent_id))
        
        self.rewards = {agent_id: 0 for agent_id in self.agents}
        self.dones = {agent_id: False for agent_id in self.agents}
        
        return {agent_id: agent.observation for agent_id, agent in self.agents.items()}
    
    def step(self, action_dict):
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        # Intended positions based on actions
        intended_position = {agent_id: agent.position for agent_id, agent in self.agents.items()}

        for agent_id, agent in self.agents.items():
            if not self.dones[agent_id]:
                action = action_dict[agent_id]
                position = agent.position

                # Movement logic
                movement_map = {
                    0: (position[0] - 1, position[1]),  # Left
                    1: (position[0] + 1, position[1]),  # Right
                    2: (position[0], position[1] + 1),  # Forward
                    3: (position[0], position[1] - 1),  # Back
                }
                new_pos = movement_map.get(action, position)
                intended_position[agent_id] = new_pos if new_pos in self.track else position

        # Collision handling and reward updates
        for agent_id, agent in self.agents.items():
            prev_pos = agent.position
            agent.position = intended_position[agent_id]

            # Reward for progressing distance
            if agent.position != prev_pos:
                agent.total_distance_traveled += 1
                agent.steps_since_last_checkpoint += 1
                agent.reward += 1  # Small reward for moving

            # Collision detection
            for other_id, other_agent in self.agents.items():
                if agent_id != other_id and agent.position == other_agent.position:
                    agent.collision_counter = 2
                    other_agent.collision_counter = 2
                    agent.reward -= 10  # Penalty for collision
                    intended_position[agent_id] = prev_pos

            # Checkpoint rewards
            if agent.position in self.checkpoints[agent.checkpoint_counters]:
                agent.checkpoint_counters += 1
                agent.steps_since_last_checkpoint = 0
                agent.reward += 50 * (1 + 1 / (agent.checkpoint_counters + 1))  # Reward diminishes with more checkpoints

                if agent.checkpoint_counters >= len(self.checkpoints):
                    agent.reward += 1000  # Bonus for completing all checkpoints
                    agent.done = True

            # Angular displacement reward
            prev_angle = agent.angle
            agent.angle = angular_displacement(prev_angle, self.grid_rows, agent.position)
            angle_smoothness = abs(agent.angle - prev_angle)
            agent.reward += max(0, 10 - angle_smoothness)  # Reward for smooth transitions

            # Penalty for idling
            agent.reward -= 1  # Small time penalty

            # Update observation
            agent.observation = self.get_observation(agent_id)
            self.dones[agent_id] = agent.done

        # Team and enemy interaction
        for agent_id, agent in self.agents.items():
            # Reward for teammate's success
            if self.dones[agent.teammate_id]:
                agent.reward += 500

            # Penalty for enemy success
            for other_id, other_agent in self.agents.items():
                if other_id != agent_id and other_id != agent.teammate_id:
                    if self.dones[other_id]:
                        agent.reward -= 500

        # Update all dictionaries
        rewards = {agent_id: agent.reward for agent_id, agent in self.agents.items()}
        observations = {agent_id: agent.observation for agent_id, agent in self.agents.items()}
        dones = self.dones
        infos = {}

        return observations, rewards, dones, infos


    def render(self):
        if self.render_mode != "human":
            return

        # Clear screen with white background
        self.screen.fill(WHITE)
        
        # Draw track
        for x, y in self.track:
            pygame.draw.rect(
                self.screen,
                GRAY,
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )
        
        # Draw start line
        for x, y in self.start_line:
            pygame.draw.rect(
                self.screen,
                GREEN,
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )
        
        # Draw checkpoints
        for i, checkpoint in enumerate(self.checkpoints):
            color = (
                int(124 + (131 * i/len(self.checkpoints))),  # R: 124 -> 255
                int(252 - (252 * i/len(self.checkpoints))),  # G: 252 -> 0
                int(0 + (255 * i/len(self.checkpoints)))     # B: 0 -> 255
            )
            for x, y in checkpoint:
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    2  # Border thickness
                )
            
            # Add checkpoint index number
            # Get the middle point of the checkpoint
            if checkpoint:  # Make sure checkpoint has points
                mid_point = checkpoint[len(checkpoint)//2]
                x, y = mid_point
                # Render the checkpoint number
                text = self.font.render(str(i), True, color)
                text_rect = text.get_rect(center=(
                    x * CELL_SIZE + CELL_SIZE/2,
                    y * CELL_SIZE + CELL_SIZE/2
                ))
                self.screen.blit(text, text_rect)
        
        # Draw cars
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'position'):  # Check if position exists
                x, y = agent.position
                color = CAR_COLORS[agent_id % len(CAR_COLORS)]
                
                # Draw car as a filled circle with a border
                center = (int(x * CELL_SIZE + CELL_SIZE/2), 
                         int(y * CELL_SIZE + CELL_SIZE/2))
                radius = int(CELL_SIZE/2) - 2
                
                # Draw filled circle
                pygame.draw.circle(self.screen, color, center, radius)
                
                # Draw border
                pygame.draw.circle(self.screen, BLACK, center, radius, 2)
                
                # Draw checkpoint counter
                if hasattr(agent, 'checkpoint_counters'):
                    text = self.font.render(str(agent.checkpoint_counters), True, BLACK)
                    text_rect = text.get_rect(center=center)
                    self.screen.blit(text, text_rect)
                
                # Draw collision indicator
                if hasattr(agent, 'collision_counter') and agent.collision_counter > 0:
                    pygame.draw.circle(
                        self.screen,
                        (255, 0, 0),  # Red
                        (center[0], center[1] - CELL_SIZE//2 - 5),
                        4
                    )
        
        # Draw grid lines
        for i in range(self.grid_rows + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen,
                (200, 200, 200),  # Light gray
                (i * CELL_SIZE, 0),
                (i * CELL_SIZE, SCREEN_SIZE),
                1
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                (200, 200, 200),  # Light gray
                (0, i * CELL_SIZE),
                (SCREEN_SIZE, i * CELL_SIZE),
                1
            )
        
        # Add info text
        y_offset = 10
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'checkpoint_counters'):
                text = f"Car {agent_id}: CP {agent.checkpoint_counters}/{len(self.checkpoints)}"
                text_surface = self.font.render(
                    text, True, CAR_COLORS[agent_id % len(CAR_COLORS)]
                )
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 25

        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS
    

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

    def get_observation(self, agent_id):

        observation = np.zeros((self.grid_rows, self.grid_columns), dtype=int)

        # 1 for track
        for (x,y) in self.track:
            observation[x, y] = 1
        
        for other_agent_id, other_agent in self.agents.items():
            pos = other_agent.position
            # If self 2
            if other_agent_id == agent_id:
                observation[pos[0], pos[1]] = 2
            else:
                # If teammate 2
                if (agent_id // 2) == (other_agent_id // 2):
                    observation[pos[0], pos[1]] = 3
                # If enemy 3
                else:
                    observation[pos[0], pos[1]] = 4

        # Get the agent's angle, teammate's angle, and angles of other agents
        agent_angle = self.agents[agent_id].angle
        teammate_angle = self.agents[self.agents[agent_id].teammate_id].angle

        # Get the angles of all other agents excluding the current agent and their teammate
        other_agents_angles = np.array([
            other_agent.angle for other_id, other_agent in self.agents.items()
            if other_id != agent_id and other_id != self.agents[agent_id].teammate_id
        ])

        observation = np.concatenate([
            observation.flatten(),           # Flattened previous observation
            np.array([agent_angle]),         # Current agent's angle
            np.array([teammate_angle]),      # Teammate's angle
            other_agents_angles              # Angles of all other agents
        ])
        
        return observation