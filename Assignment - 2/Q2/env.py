import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
TILE_DIMENSION = 50
map_layout = [
    ['#', '#', '#', '#', '#', '#'],
    ['#', ' ', '@', '#', '#', '#'],
    ['#', ' ', ' ', '#', '#', '#'],
    ['#', '.', ' ', ' ', ' ', '#'],
    ['#', ' ', ' ', '$', ' ', '#'],
    ['#', ' ', ' ', '#', '#', '#'],
    ['#', '#', '#', '#', '#', '#'],
]

# Calculate grid size based on the map layout
GRID_ROWS = len(map_layout)
GRID_COLUMNS = len(map_layout[0])

SCREEN_WIDTH = GRID_COLUMNS * TILE_DIMENSION
SCREEN_HEIGHT = GRID_ROWS * TILE_DIMENSION
FRAMERATE = 60

# Define Colors
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
BLUE_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BROWN_COLOR = (139, 69, 19)

# Symbols for different game elements
BLOCK = '#'
EMPTY = ' '
BOX = '$'
TARGET = '.'
PLAYER = '@'
PLAYER_ON_TARGET = '+'
BOX_ON_TARGET = '*'

# Set up the game screen
window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Sokoban Game')

# Frame rate control
timer = pygame.time.Clock()

# Get player starting position
def find_player_position():
    for r in range(GRID_ROWS):
        for c in range(GRID_COLUMNS):
            if map_layout[r][c] in (PLAYER, PLAYER_ON_TARGET):
                return (r, c)
    return None

# Player movement logic
def handle_player_movement(row_delta, col_delta):
    global map_layout
    current_row, current_col = find_player_position()

    next_row = current_row + row_delta
    next_col = current_col + col_delta
    beyond_next_row = next_row + row_delta
    beyond_next_col = next_col + col_delta

    # Check if the movement is within grid limits
    if not (0 <= next_row < GRID_ROWS) or not (0 <= next_col < GRID_COLUMNS):
        return

    # Move player if stepping on an empty space or target
    if map_layout[next_row][next_col] in (EMPTY, TARGET):
        if map_layout[current_row][current_col] == PLAYER_ON_TARGET:
            map_layout[current_row][current_col] = TARGET
        else:
            map_layout[current_row][current_col] = EMPTY

        map_layout[next_row][next_col] = PLAYER_ON_TARGET if map_layout[next_row][next_col] == TARGET else PLAYER

    # Move the crate if the next position contains one
    elif map_layout[next_row][next_col] in (BOX, BOX_ON_TARGET):
        if not (0 <= beyond_next_row < GRID_ROWS) or not (0 <= beyond_next_col < GRID_COLUMNS):
            return

        # Move the crate if the space beyond is empty or a target
        if map_layout[beyond_next_row][beyond_next_col] in (EMPTY, TARGET):
            if map_layout[current_row][current_col] == PLAYER_ON_TARGET:
                map_layout[current_row][current_col] = TARGET
            else:
                map_layout[current_row][current_col] = EMPTY

            map_layout[next_row][next_col] = PLAYER_ON_TARGET if map_layout[next_row][next_col] == BOX_ON_TARGET else PLAYER
            map_layout[beyond_next_row][beyond_next_col] = BOX_ON_TARGET if map_layout[beyond_next_row][beyond_next_col] == TARGET else BOX

# Function to render the game elements
def render_grid():
    window.fill(WHITE_COLOR)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLUMNS):
            x_coord = c * TILE_DIMENSION
            y_coord = r * TILE_DIMENSION

            if map_layout[r][c] == BLOCK:
                pygame.draw.rect(window, BLACK_COLOR, (x_coord, y_coord, TILE_DIMENSION, TILE_DIMENSION))
            elif map_layout[r][c] in (EMPTY, TARGET, PLAYER, BOX, PLAYER_ON_TARGET, BOX_ON_TARGET):
                pygame.draw.rect(window, BROWN_COLOR, (x_coord, y_coord, TILE_DIMENSION, TILE_DIMENSION))

            if map_layout[r][c] == PLAYER:
                pygame.draw.circle(window, BLUE_COLOR, (x_coord + TILE_DIMENSION // 2, y_coord + TILE_DIMENSION // 2), TILE_DIMENSION // 3)
            elif map_layout[r][c] == BOX:
                pygame.draw.rect(window, GREEN_COLOR, (x_coord + 10, y_coord + 10, TILE_DIMENSION - 20, TILE_DIMENSION - 20))
            elif map_layout[r][c] == TARGET:
                pygame.draw.circle(window, BLACK_COLOR, (x_coord + TILE_DIMENSION // 2, y_coord + TILE_DIMENSION // 2), TILE_DIMENSION // 6)
            elif map_layout[r][c] == BOX_ON_TARGET:
                pygame.draw.circle(window, BLACK_COLOR, (x_coord + TILE_DIMENSION // 2, y_coord + TILE_DIMENSION // 2), TILE_DIMENSION // 6)
                pygame.draw.rect(window, GREEN_COLOR, (x_coord + 10, y_coord + 10, TILE_DIMENSION - 20, TILE_DIMENSION - 20))
            elif map_layout[r][c] == PLAYER_ON_TARGET:
                pygame.draw.circle(window, BLACK_COLOR, (x_coord + TILE_DIMENSION // 2, y_coord + TILE_DIMENSION // 2), TILE_DIMENSION // 6)
                pygame.draw.circle(window, BLUE_COLOR, (x_coord + TILE_DIMENSION // 2, y_coord + TILE_DIMENSION // 2), TILE_DIMENSION // 3)

# Check if the player has completed the level
def has_won():
    for r in range(GRID_ROWS):
        for c in range(GRID_COLUMNS):
            if map_layout[r][c] == BOX:
                return False
    return True

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                handle_player_movement(-1, 0)
            elif event.key == pygame.K_DOWN:
                handle_player_movement(1, 0)
            elif event.key == pygame.K_LEFT:
                handle_player_movement(0, -1)
            elif event.key == pygame.K_RIGHT:
                handle_player_movement(0, 1)

    render_grid()

    if has_won():
        print("Congratulations! You completed the level!")
        pygame.quit()
        sys.exit()

    pygame.display.update()
    timer.tick(FRAMERATE)
