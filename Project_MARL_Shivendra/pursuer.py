from player import *
import random
import math
import pickle
from policy import *

# initialise pursuer
class Pursuer(Player):
    initial_positions = [(280, 250), (400, 400), (250, 300), (500, 100), (620, 450)]

    def __init__(self):
        # super().__init__()
        rect = pygame.Rect(0, 0, 32, 32)
        rect.center = random.choice(Pursuer.initial_positions)
        self.rect = rect
        self.original_cords = [self.rect.x, self.rect.y]
        self.angle = 0
        self.checkwalls(850)
        self.orientation = 0
        self.movement_type = [1, 3][0]
        self.game_no = 0
        self.game_prevno = 0
        self.dist_covered = 0
        self.agent_pursuer = Policy(self)
        pursuer_pickle = open("pursuer_qtable.pickle","rb")
        self.agent_pursuer.q_table=pickle.load(pursuer_pickle)

    def distance_reward(self, evader_cords):
        x = self.rect.x
        y = self.rect.y
        for i in evader_cords:
            x1 = [x, y]
            x2 = [i[0], i[1]]
            dist = math.dist(x1, x2)
            if dist > 100:
                return -0.01
            else:
                return dist / 1000

    def area_coverage(self):
        # function for rewarding when more area is covered
        self.dist_covered = self.distance()
        if self.dist_covered < 100:
            coverage_reward = - self.dist_covered / 1000
        else:
            coverage_reward = self.dist_covered / 1000
        return coverage_reward
    
    def distance(self):
        # function for rewarding when near evaders
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

    def reward(self, evader_objs, evader_cords):
        reward = -1
        co_list = []
        self.v_startpoints, self.v_endpoints = Vision(45,180,50).get_intersect(self.rect.center, self.near_walls,
                                                                      self.orientation)
        
        loser_evader = 0
        pursuer_rew=0
        for e in evader_objs:
            for line in self.vision: 
                start = e.rect.clipline(line)
                if start:
                    pursuer_rew = pursuer_rew + 1000
                    # print("They collided!!!!!!!!!!! ", pursuer_rew)
                    loser_evader = e
                    self.game_prevno = self.game_no
                    self.game_no += 1
                    break
        reward = pursuer_rew # if pursuer spots any evaders.
        reward += self.distance_reward(evader_cords)  # distance from evader, if more negative reward, if less, positive reward
        reward += self.area_coverage()  # distance from its own initial cords reward function
        reward += self.wall_collision()  # add reward func for continuous collisions with walls if needed ... using is_walls_nearby
        # print(reward)
        return reward, loser_evader