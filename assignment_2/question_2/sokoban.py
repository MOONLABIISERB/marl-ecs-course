import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
TILE_SIZE = 50
level = [
    ['#', '#', '#', '#', '#', '#'],
    ['#', ' ', '@', '#', '#', '#'],
    ['#', ' ', ' ', '#', '#', '#'],
    ['#', '.', ' ', ' ', ' ', '#'],
    ['#', ' ', ' ', '$', ' ', '#'],
    ['#', ' ', ' ', '#', '#', '#'],
    ['#', '#', '#', '#', '#', '#'],
]

# Dynamically calculate grid dimensions based on the level array
GRID_HEIGHT = len(level)
GRID_WIDTH = len(level[0])

WINDOW_WIDTH = GRID_WIDTH * TILE_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * TILE_SIZE
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BROWN = (139, 69, 19)

# Game entities
WALL = '#'
FLOOR = ' '
CRATE = '$'
GOAL = '.'
PLAYER = '@'
PLAYER_ON_GOAL = '+'
CRATE_ON_GOAL = '*'

# Initialize screen
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Sokoban')

# Clock to control the frame rate
clock = pygame.time.Clock()

# Load player position
def get_player_pos():
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            if level[row][col] == PLAYER or level[row][col] == PLAYER_ON_GOAL:
                return (row, col)
    return None

# Movement function
def move_player(drow, dcol):
    global level
    player_row, player_col = get_player_pos()

    new_row = player_row + drow
    new_col = player_col + dcol
    beyond_new_row = new_row + drow
    beyond_new_col = new_col + dcol

    # Check if the new position is inside the grid bounds
    if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
        return

    # Check if the new position is either floor or goal
    if level[new_row][new_col] in (FLOOR, GOAL):  # Move player
        if level[player_row][player_col] == PLAYER_ON_GOAL:
            level[player_row][player_col] = GOAL
        else:
            level[player_row][player_col] = FLOOR

        if level[new_row][new_col] == GOAL:
            level[new_row][new_col] = PLAYER_ON_GOAL
        else:
            level[new_row][new_col] = PLAYER

    # Check if the new position contains a crate
    elif level[new_row][new_col] in (CRATE, CRATE_ON_GOAL):
        if beyond_new_row < 0 or beyond_new_row >= GRID_HEIGHT or beyond_new_col < 0 or beyond_new_col >= GRID_WIDTH:
            return

        # Move crate if the space beyond is floor or goal
        if level[beyond_new_row][beyond_new_col] in (FLOOR, GOAL):
            if level[player_row][player_col] == PLAYER_ON_GOAL:
                level[player_row][player_col] = GOAL
            else:
                level[player_row][player_col] = FLOOR

            if level[new_row][new_col] == CRATE_ON_GOAL:
                level[new_row][new_col] = PLAYER_ON_GOAL
            else:
                level[new_row][new_col] = PLAYER

            if level[beyond_new_row][beyond_new_col] == GOAL:
                level[beyond_new_row][beyond_new_col] = CRATE_ON_GOAL
            else:
                level[beyond_new_row][beyond_new_col] = CRATE

# Draw the game entities
def draw_level():
    screen.fill(WHITE)

    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            x = col * TILE_SIZE
            y = row * TILE_SIZE

            if level[row][col] == WALL:
                pygame.draw.rect(screen, BLACK, (x, y, TILE_SIZE, TILE_SIZE))
            elif level[row][col] in (FLOOR, GOAL, PLAYER, CRATE, PLAYER_ON_GOAL, CRATE_ON_GOAL):
                pygame.draw.rect(screen, BROWN, (x, y, TILE_SIZE, TILE_SIZE))

            if level[row][col] == PLAYER:
                pygame.draw.circle(screen, BLUE, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), TILE_SIZE // 3)
            elif level[row][col] == CRATE:
                pygame.draw.rect(screen, GREEN, (x + 10, y + 10, TILE_SIZE - 20, TILE_SIZE - 20))
            elif level[row][col] == GOAL:
                pygame.draw.circle(screen, BLACK, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), TILE_SIZE // 6)
            elif level[row][col] == CRATE_ON_GOAL:
                pygame.draw.circle(screen, BLACK, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), TILE_SIZE // 6)
                pygame.draw.rect(screen, GREEN, (x + 10, y + 10, TILE_SIZE - 20, TILE_SIZE - 20))
            elif level[row][col] == PLAYER_ON_GOAL:
                pygame.draw.circle(screen, BLACK, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), TILE_SIZE // 6)
                pygame.draw.circle(screen, BLUE, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), TILE_SIZE // 3)

# Check if the player has won
def check_win():
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            if level[row][col] == CRATE:
                return False
    return True


print(level)


# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Handle keypresses more smoothly using KEYDOWN events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                move_player(-1, 0)
            elif event.key == pygame.K_DOWN:
                move_player(1, 0)
            elif event.key == pygame.K_LEFT:
                move_player(0, -1)
            elif event.key == pygame.K_RIGHT:
                move_player(0, 1)

    draw_level()

    if check_win():
        print("You won!")
        pygame.quit()
        sys.exit()

    pygame.display.update()
    clock.tick(FPS)
