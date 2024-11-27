import random
import math


def to_normal(point, height=500):
    return point[0], height - point[1]


def create_points(theta, radius, density):
    points = []
    c = 0
    while c < theta:
        points.append((radius * math.cos(math.radians(c)), radius * math.sin(math.radians(c))))
        c += theta / density

    return points
    pass

def point_translation(origin, end_points):
    mx, my = origin
    ex, ey = end_points
    return ex + mx, ey + my
    pass


def point_rotation(origin, point, angle):
    ox, oy = to_normal(origin)
    px, py = to_normal(point)
    angle = math.radians(angle)
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return to_normal((qx, qy))
    pass


walls = []


class Vision:

    def __init__(self,angle,radius,density):
        self.angle=angle
        self.radius=radius
        self.density=density
        self.pos = (0, 0)
        self.walls = []

    def get_lines(self, pos, walls, angle):
        self.lines = []
        for point in create_points(self.angle, self.radius, self.density):
            mx, my = pos
            point = point_translation((mx, my), point)
            point = point_rotation((mx, my), point, self.angle / 2)
            point = point_rotation((mx, my), point, angle)
            px, py = point
            line = (mx, my, px, py)
            for wall in walls:
                clipped_line = wall.rect.clipline(line)
                if clipped_line:
                    start, end = clipped_line
                    x1, y1 = start
                    line = (mx, my, x1, y1)
            self.lines.append(line)
        return self.lines

    def get_intersect(self, pos, walls, angle):
        self.start_lines = []
        self.end_lines = []
        for point in create_points(self.angle, self.radius, self.density):
            mx, my = pos
            point = point_translation((mx, my), point)
            point = point_rotation((mx, my), point, self.angle / 2)
            point = point_rotation((mx, my), point, angle)
            px, py = point
            line = (mx, my, px, py)
            start_ll = (mx,my)
            self.start_lines.append(start_ll)
            end_ll = (px, py)
            for wall in walls:
                clipped_line = wall.rect.clipline(line)
                if clipped_line:
                    start, end = clipped_line
                    x1, y1 = start
                    end_ll = (x1, y1)
            self.end_lines.append(end_ll)
        return self.start_lines, self.end_lines

    pass
