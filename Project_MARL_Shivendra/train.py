import os
from policy import *
from raycast import *
from level import *
from colors import *
from walls import *
from pursuer import *
from evader import *
import pygame
import pickle
import pygame.freetype 
import tkinter as tk

root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()

width = 1500
height = 800

size = [width, height]
# print(size)
pygame.init()

screen = pygame.display.set_mode(size)

# Initialise pygame
os.environ["SDL_VIDEO_CENTERED"] = "0"
pygame.init()

GAME_FONT = pygame.freetype.Font("font.ttf", 16)
GAME_FONT_LARGE = pygame.freetype.Font("font.ttf", 24)

# Set up the display
pygame.display.set_caption("Pursuit & Evasion")

clock = pygame.time.Clock()

parse_level(level)
esc = 0
episodes = 0
number_of_episodes = 5000


logs = {
    "episodes" : [],
    "purser_reward" : [],
    "evader_reward" : [],
}


while episodes < number_of_episodes and esc == 0:

    evader1 = Evader()  
    evader2 = Evader()
    pursuer1 = Pursuer()

    evader_objs = [evader1, evader2]
    pursuer_objs = [pursuer1]
    running = True
    x = 0
    while running:

        clock.tick(60)
        # print(width, height)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
                esc = 1
            if e.type == pygame.KEYDOWN:
                key = pygame.key.get_pressed()
            

        action_list_evader = []
        for e in evader_objs:
            evader_direction, evader_rotation = e.agent_evader.get_action()
            e.act(evader_direction, evader_rotation)
            action_list_evader.append([evader_direction, evader_rotation])

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

        # Draw the scene
        screen.fill((0, 0, 0))
        floor = pygame.image.load('floor.jpg', "r")
        screen.blit(floor, (800, 400))
        floor = pygame.image.load('sky.jpg', "r")
        screen.blit(floor, (800, 0))

        for wall in walls:
            pygame.draw.rect(screen, white, wall.rect)

        for e in evader_objs:
            for line in e.vision:
                mx, my, px, py = line
                # pygame.draw.aaline(screen, light_green, (mx, my), (px, py))

        for p in pursuer_objs:
            for line in pursuer1.vision:
                mx, my, px, py = line
                # pygame.draw.aaline(screen, light_purple, (mx, my), (px, py))

        text_surface, rect = GAME_FONT.render("PURSUIT & EVASION", light_green)
        screen.blit(text_surface, (100, 818))
        
        text_surface, rect = GAME_FONT.render(f"Number of Episodes Completed  = {episodes}", cyan)
        screen.blit(text_surface, (350, 818))
        
        temp = 0
        for e in evader_objs:
            text_surface, rect = GAME_FONT.render(f"Evader Rewards are {str(e.agent_evader.total_reward)[:6]}",
                                                  forest_green)
            screen.blit(text_surface, (700, 818 + temp))
            temp += 20

        for p in pursuer_objs:
            text_surface, rect = GAME_FONT.render(f"Pursuer Rewards are {str(p.agent_pursuer.total_reward)[:6]}", purple)
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

        if not evader_objs:
            logs['episodes'].append(episodes)
            logs['evader_reward'].append(p.agent_pursuer.total_reward)
            logs['purser_reward'].append(e.agent_evader.total_reward)
            print(f"Episodes {episodes + 1} completed!")
            episodes += 1
            running = False

        if episodes == number_of_episodes:
            for p in pursuer_objs:
                pickle_p = open("pursuer_qtable.pickle", "wb")
                pickle.dump(p.agent_pursuer.q_table, pickle_p)
                pickle_p.close()

            for e in evader_objs:
                pickle_e = open("evader_qtable.pickle", "wb")
                pickle.dump(e.agent_evader.q_table, pickle_e)
                pickle_e.close()
                
            with open("game_logs.pickle", "wb") as pickle_logs:
                pickle.dump(logs, pickle_logs)

            print("Pickle updated and logs saved")

        pygame.display.flip()
