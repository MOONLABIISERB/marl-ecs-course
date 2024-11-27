import os
import cv2
import pygame
import pickle
import tkinter as tk
from pursuer import *
from evader import *
from raycast import *
from level import *
from walls import *
from colors import *
from policy import *
import pygame.freetype

# Initialize pygame
pygame.init()

# Define fonts
GAME_FONT = pygame.freetype.Font("font.ttf", 16)
GAME_FONT_LARGE = pygame.freetype.Font("font.ttf", 24)

root = tk.Tk()
width = 1400
height = 800
size = [width, height]
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Pursuit & Evasion - Test")

# Load the Q-tables
with open("pursuer_qtable.pickle", "rb") as f:
    pursuer_qtable = pickle.load(f)

with open("evader_qtable.pickle", "rb") as f:
    evader_qtable = pickle.load(f)

# Initialize the agents with the loaded Q-tables
evader1 = Evader()
evader2 = Evader()
# evader3 = Evader()
# evader4 = Evader()
# evader5 = Evader()
pursuer1 = Pursuer()

evader_objs = [evader1, evader2]
# evader_objs = [evader1, evader2, evader3, evader4, evader5]

pursuer_objs = [pursuer1]

# Initialize the level
parse_level(level)  

fourcc = cv2.VideoWriter_fourcc(*'XVID')  
video_filename = 'simulation_output.avi' 
out = cv2.VideoWriter(video_filename, fourcc, 60.0, (width, height))

clock = pygame.time.Clock()

running = True
episodes = 0
max_steps = 100
x = 0

while running and episodes < 1: 
    for e in evader_objs:
        e.agent_evader.q_table = evader_qtable  
    for p in pursuer_objs:
        p.agent_pursuer.q_table = pursuer_qtable  
    
    steps = 0
    while steps < max_steps:
        clock.tick(60) 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        # Action for evaders
        action_list_evader = []
        for e in evader_objs:
            evader_direction, evader_rotation = e.agent_evader.get_action()
            e.act(evader_direction, evader_rotation)
            action_list_evader.append([evader_direction, evader_rotation])

        # Action for pursuers
        action_list_pursuer = []
        for p in pursuer_objs:
            pursuer_direction, pursuer_rotation = p.agent_pursuer.get_action()
            p.act(pursuer_direction, pursuer_rotation)
            action_list_pursuer.append([pursuer_direction, pursuer_rotation])

        evader_cords = get_cords(evader_objs)
        pursuer_cords = get_cords(pursuer_objs)

        evader_temp = []
        for e in evader_objs:
            evader_temp = action_list_evader.pop(0)
            evader_direction = evader_temp[0]
            evader_reward = e.reward(pursuer_objs, pursuer_cords)
            e.agent_evader.update(evader_direction, evader_reward)

        pursuer_temp = []
        for p in pursuer_objs:
            pursuer_temp = action_list_pursuer.pop(0)
            pursuer_direction = pursuer_temp[0]
            pursuer_reward, catch = pursuer1.reward(evader_objs, evader_cords)
            if catch:
                evader_objs.remove(catch)
                del catch
            pursuer1.agent_pursuer.update(pursuer_direction, pursuer_reward)

        screen.fill((0, 0, 0))
        floor = pygame.image.load('floor.jpg', "r")
        screen.blit(floor, (800, 400))
        floor = pygame.image.load('sky.jpg', "r")
        screen.blit(floor, (800, 0))

        for wall in walls:
            pygame.draw.rect(screen, white, wall.rect)

        for e in evader_objs:
            pygame.draw.rect(screen, forest_green, e.rect)
        for p in pursuer_objs:
            pygame.draw.rect(screen, purple, p.rect)

        text_surface, rect = GAME_FONT.render("PURSUIT & EVASION - TEST", light_green)
        screen.blit(text_surface, (100, 818))

        temp = 0
        for e in evader_objs:
            text_surface, rect = GAME_FONT.render(f"Evader Rewards: {str(e.agent_evader.total_reward)[:6]}", forest_green)
            screen.blit(text_surface, (700, 818 + temp))
            temp += 20

        for p in pursuer_objs:
            text_surface, rect = GAME_FONT.render(f"Pursuer Rewards: {str(p.agent_pursuer.total_reward)[:6]}", purple)
            screen.blit(text_surface, (1000, 818))

        for e in evader_objs:
            pygame.draw.rect(screen, forest_green, e.rect)
        for p in pursuer_objs:
            pygame.draw.rect(screen, purple, p.rect)

        agent_lines = []
        for e in evader_objs:
            e_lines = Raycast(e).get_lines()
            agent_lines.append(e_lines)
            
        for p in pursuer_objs:
            p_lines = Raycast(p).get_lines()
            agent_lines.append(p_lines)

        px = 0
        color = (3, 73, 252)
        for l in agent_lines[x]:
            if l['orientation'] == 'v':
                color = (168, 168, 168)
                pygame.draw.line(screen, color, (800 + px, 400 + l['length'] / 2),
                                 (800 + px, 400 - l['length'] + l['length'] / 2), width=3)
                px += 2
                pygame.draw.line(screen, (0, 0, 0), (800 + px, 400 + l['length'] / 2),
                                 (800 + px, 400 - l['length'] + l['length'] / 2), width=1)
                px += 2
            elif l['orientation'] == 'h':
                color = (107, 107, 107)
                pygame.draw.line(screen, color, (800 + px, 400 + l['length'] / 2),
                                 (800 + px, 400 - l['length'] + l['length'] / 2), width=3)
                px += 2
                pygame.draw.line(screen, (0, 0, 0), (800 + px, 400 + l['length'] / 2),
                                 (800 + px, 400 - l['length'] + l['length'] / 2), width=1)
                px += 2
            elif l['orientation'] == 'e':  # edge
                color = (0, 0, 0)
                pygame.draw.line(screen, color, (800 + px, 400 + l['length'] / 2),
                                 (800 + px, 400 - l['length'] + l['length'] / 2), width=3)
                px += 2
                pygame.draw.line(screen, (0, 0, 0), (800 + px, 400 + l['length'] / 2),
                                 (800 + px, 400 - l['length'] + l['length'] / 2), width=1)
                px += 2
            elif l['orientation'] == 'i':  # inf
                px += 4
                pass

        if pursuer1.game_no == pursuer1.game_prevno + 1:
            pursuer1.game_prevno = pursuer1.game_no

        pygame.display.flip()

        frame = pygame.surfarray.array3d(screen)
        frame = frame.swapaxes(0, 1)  
        out.write(frame)  

        # Check if evaders are caught
        if not evader_objs:
            print("Evaders caught!")
            running = False
            episodes += 1
            break
        steps += 1

out.release()
pygame.quit()