import math


class Raycast:
    max_height = 400

    def __init__(self, player):
        self.player = player
        pass

    def x_or_y(self):
        for line in self.player.render:
            a, b, x, y = line

    def get_lines(self):
        lines = list()
        lengths = []
        orientations = ''  # will use str function to fix oddities in colors
        o = ''
        last_length = 100
        for line in self.player.render:
            a, b, x, y = line
            x = int(x)
            y = int(y)
            if x % 32 == 31 or x % 32 == 0:
                o = 'v'
            elif y % 32 == 31 or y % 32 == 0:
                o = 'h'
            d = max(0.01, math.dist((a, b), (x, y)))
            r = 400 / d * 250
            if 0.99 < d / 800 < 1.01:
                o = 'i'
            lengths.append(min(750, r))
            orientations = orientations + o

            pass
        orientations = orientations.replace('hvh', 'hhh')
        orientations = orientations.replace('vhv', 'vvv')
        orientations = orientations.replace('hvvh', 'hhhh')
        orientations = orientations.replace('vhhv', 'vvvv')
        orientations = orientations.replace('hvvvh', 'hhhhh')
        orientations = orientations.replace('vhhhv', 'vvvvv')
        orientations = orientations.replace('hvvvvh', 'hhhhhh')
        orientations = orientations.replace('vhhhhv', 'vvvvvv')  # remove anamolies
        orientations = orientations.replace('vh', 've')
        orientations = orientations.replace('hv', 'he')

        orientations = list(orientations)
        for c in range(len(orientations)):
            lines.append({'orientation': orientations[c], 'length': lengths[c]})
        return lines
