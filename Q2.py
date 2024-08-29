import numpy as np
import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 9
CELL_SIZE = 60
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Create the window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Value iteration")

class GridWorld:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.start = (8, 0)
        self.goal = (0, 8)
        self.tunnel_in = (6, 2)
        self.tunnel_out = (2, 6)
        self.walls = [(7, 3), (6, 3), (5, 3), (5, 2), (5, 1), (1, 5), (2, 5), (3, 5), (3, 6), (3, 7), (0, 5), (3,8)]
        self.actions = ['up', 'down', 'left', 'right']

        # Mark walls in the grid
        for wall in self.walls:
            self.grid[wall] = 1

    def get_next_state(self, state, action):
        x, y = state
        if state == self.tunnel_in:
            return self.tunnel_out
        if action == 'up':
            next_state = (max(x-1, 0), y)
        elif action == 'down':
            next_state = (min(x+1, GRID_SIZE-1), y)
        elif action == 'left':
            next_state = (x, max(y-1, 0))
        elif action == 'right':
            next_state = (x, min(y+1, GRID_SIZE-1))
        
        # Check if next state is a wall
        if next_state in self.walls:
            return state  # Stay in the current state if hitting a wall
        return next_state

    def get_reward(self, state):
        return 1 if state == self.goal else 0

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    while True:
        delta = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) in env.walls:
                    continue  # Skip walls
                v = V[i, j]
                max_v = max([env.get_reward((i,j)) + gamma * V[env.get_next_state((i,j), a)] for a in env.actions])
                V[i, j] = max_v
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break
    
    policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) in env.walls:
                policy[i, j] = -1  # No action for walls
            else:
                policy[i, j] = np.argmax([V[env.get_next_state((i,j), a)] for a in env.actions])
    
    return V, policy

def draw_grid(surface):
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(surface, BLACK, (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(surface, BLACK, (0, y), (WINDOW_SIZE, y))

def draw_cell(surface, x, y, color):
    pygame.draw.rect(surface, color, (y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE))

def draw_arrow(surface, x, y, direction):
    center = (y*CELL_SIZE + CELL_SIZE//2, x*CELL_SIZE + CELL_SIZE//2)
    if direction == 0:  # Up
        pygame.draw.line(surface, RED, center, (center[0], center[1] - CELL_SIZE//3), 3)
        pygame.draw.line(surface, RED, (center[0], center[1] - CELL_SIZE//3), (center[0] - CELL_SIZE//6, center[1] - CELL_SIZE//6), 3)
        pygame.draw.line(surface, RED, (center[0], center[1] - CELL_SIZE//3), (center[0] + CELL_SIZE//6, center[1] - CELL_SIZE//6), 3)
    elif direction == 1:  # Down
        pygame.draw.line(surface, RED, center, (center[0], center[1] + CELL_SIZE//3), 3)
        pygame.draw.line(surface, RED, (center[0], center[1] + CELL_SIZE//3), (center[0] - CELL_SIZE//6, center[1] + CELL_SIZE//6), 3)
        pygame.draw.line(surface, RED, (center[0], center[1] + CELL_SIZE//3), (center[0] + CELL_SIZE//6, center[1] + CELL_SIZE//6), 3)
    elif direction == 2:  # Left
        pygame.draw.line(surface, RED, center, (center[0] - CELL_SIZE//3, center[1]), 3)
        pygame.draw.line(surface, RED, (center[0] - CELL_SIZE//3, center[1]), (center[0] - CELL_SIZE//6, center[1] - CELL_SIZE//6), 3)
        pygame.draw.line(surface, RED, (center[0] - CELL_SIZE//3, center[1]), (center[0] - CELL_SIZE//6, center[1] + CELL_SIZE//6), 3)
    elif direction == 3:  # Right
        pygame.draw.line(surface, RED, center, (center[0] + CELL_SIZE//3, center[1]), 3)
        pygame.draw.line(surface, RED, (center[0] + CELL_SIZE//3, center[1]), (center[0] + CELL_SIZE//6, center[1] - CELL_SIZE//6), 3)
        pygame.draw.line(surface, RED, (center[0] + CELL_SIZE//3, center[1]), (center[0] + CELL_SIZE//6, center[1] + CELL_SIZE//6), 3)

def draw_environment(env, policy):
    screen.fill(WHITE)
    draw_grid(screen)
    
    # Draw walls
    for wall in env.walls:
        draw_cell(screen, wall[0], wall[1], GRAY)
    
    # Draw start
    draw_cell(screen, env.start[0], env.start[1], BLUE)
    
    # Draw goal
    draw_cell(screen, env.goal[0], env.goal[1], GREEN)
    
    # Draw tunnels
    draw_cell(screen, env.tunnel_in[0], env.tunnel_in[1], YELLOW)
    draw_cell(screen, env.tunnel_out[0], env.tunnel_out[1], YELLOW)
    
    # Draw policy arrows
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) not in [env.start, env.goal, env.tunnel_in, env.tunnel_out] and (i, j) not in env.walls:
                draw_arrow(screen, i, j, policy[i, j])
    
    pygame.display.flip()

def draw_policy(env, policy):
    policy_surface = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Policy Iteration")
    policy_surface.fill(WHITE)
    draw_grid(policy_surface)
    
    # Draw walls
    for wall in env.walls:
        draw_cell(policy_surface, wall[0], wall[1], GRAY)
    
    # Draw start
    draw_cell(policy_surface, env.start[0], env.start[1], BLUE)
    
    # Draw goal
    draw_cell(policy_surface, env.goal[0], env.goal[1], GREEN)
    
    # Draw tunnels
    draw_cell(policy_surface, env.tunnel_in[0], env.tunnel_in[1], YELLOW)
    draw_cell(policy_surface, env.tunnel_out[0], env.tunnel_out[1], YELLOW)
    
    # Draw policy arrows
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) not in [env.start, env.goal, env.tunnel_in, env.tunnel_out] and (i, j) not in env.walls:
                draw_arrow(policy_surface, i, j, policy[i, j])
    
    pygame.display.flip()
    
    # Keep the window open until closed by the user
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    pygame.quit()

def main():
    env = GridWorld()
    _, policy = value_iteration(env)
    
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        draw_environment(env, policy)
        clock.tick(FPS)
    
    pygame.quit()
    draw_policy(env, policy)
    sys.exit()

if __name__ == "__main__":
    main()
