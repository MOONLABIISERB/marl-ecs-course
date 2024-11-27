from player import *
import random
import math
from policy import *
import pickle


# initialise evader
class Evader(Player):
    initial_positions = [(128, 128), (600, 128), (500, 100), (400, 200), (128, 240)]

    def __init__(self):
        # super().__init__()
        rect = pygame.Rect(0, 0, 32, 32)
        rect.center = random.choice(Evader.initial_positions)
        self.rect = rect
        self.original_cords = [self.rect.x, self.rect.y]
        self.angle = 0
        self.checkwalls(850)
        self.orientation = 0
        self.movement_type = [1, 3][0]
        self.dist_covered = 0
        self.agent_evader = Policy(self)
        evader_pickle = open("evader_qtable.pickle","rb")
        self.agent_evader.q_table=pickle.load(evader_pickle)

    def distance_reward(self, pursuer_cords):
        x = self.rect.x
        y = self.rect.y
        for i in pursuer_cords:
            x1 = [x, y]
            x2 = [i[0], i[1]]
            dist = math.dist(x1, x2)
            if dist > 100:
                return dist / 1000
            else:
                return -0.01

    def area_coverage(self):
        # function for rewarding when more area is covered
        self.dist_covered = self.distance()
        if self.dist_covered < 100:
            coverage_reward = - self.dist_covered / 1000
        else:
            coverage_reward = self.dist_covered / 1000
        return coverage_reward

    def distance(self):
        # function for rewarding when more distance from puruser
        x1 = self.original_cords
        x2 = [self.rect.x, self.rect.y]
        self.dist_covered = math.dist(x1, x2)
        return self.dist_covered

    def wall_collision(self):
        wall_info = self.is_wall_nearby()
        wall_reward = 0
        for key in wall_info:
            if wall_info[key]:
                wall_reward = -0.001
        return wall_reward

    def reward(self, pursuer_objs, pursuer_cords):
        reward = -1
        co_list = []
        flag = 0
        self.v_startpoints, self.v_endpoints = Vision(45,180,50).get_intersect(self.rect.center, self.near_walls,
                                                                      self.orientation)
       
        pursuer_rew=0
        for p in pursuer_objs:
            for line in self.vision: 
                start = p.rect.clipline(line)
                if start:
                    pursuer_rew = pursuer_rew -200
                    # print("pursuer is near, run away ", pursuer_rew)
                    break
        reward = pursuer_rew # if evaders spots pursuer.

        reward += self.distance_reward(pursuer_cords)  # distance from pursuer is more then positive reward, if less then negative reward
        reward += self.area_coverage()  # distance from its own initial cords reward function
        reward += self.wall_collision()  # add reward func for continuous collisions with walls if needed  ... using is_walls_nearby
        return reward
