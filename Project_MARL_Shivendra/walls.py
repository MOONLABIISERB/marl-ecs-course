import pygame

walls = []  # List to hold the walls


# Nice class to hold a wall rect
class Wall(object):
    def __init__(self, pos):
        walls.append(self)
        self.rect = pygame.Rect(pos[0], pos[1], 32, 32)

    # Parse the level string above. W = wall, E = exit


def parse_level(level):
    x = y = 0
    for row in level:
        for col in row:
            if col == "W":
                Wall((x, y))
            if col == "E":
                end_rect = pygame.Rect(x, y, 32, 32)
            x += 32
        y += 32
        x = 0

def get_cords(objects):
        colist = []
        for i in objects:
            ll = (i.rect.x, i.rect.y)
            colist.append(ll)
        return colist

def rot_center(image, rect, angle):
    """rotate an image while keeping its center"""
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = rot_image.get_rect(center=rect.center)
    return rot_image, rot_rect
