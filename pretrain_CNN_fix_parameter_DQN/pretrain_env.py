"""
This part of code is the environment.
Using Tensorflow to build the neural network.
"""

import numpy as np
import tensorflow as tf
import pygame
from random import uniform

FPS = 90
SCREEN_WHIDTH = 672
SCREEN_HEIGHT = 672

# init the game
pygame.init()
FPSCLOCK = pygame.time.Clock()
screen = pygame.display.set_mode([SCREEN_WHIDTH, SCREEN_HEIGHT])
pygame.display.set_caption('hunting')

# load resources
background = (255, 255, 255)  # white
hunter_color = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]  # B #R #G #Y
escaper_color = (0, 0, 0)  # bck


class ENV:
    def __init__(self):
        self.hunter_radius = 8
        self.escaper_radius = 8
        self.max_pos = np.array([SCREEN_WHIDTH, SCREEN_HEIGHT])
        self.catch_angle_max = np.pi * 3 / 4  # 135°
        self.catch_dis = 50.
        self.collide_min = self.hunter_radius + self.escaper_radius + 2.
        # the center pos, x : [0, SCREEN_WHIDTH], y: [0, SCREEN_HEIGHT]
        self.delta_t = 0.1  # 100ms
        self.hunter_acc = 20
        self.escaper_acc = 10
        self.hunter_spd_max = 100  # 5 pixels once
        self.escaper_spd_max = 70
        self.hunter_spd = np.zeros([4, 2], dtype=np.float32)
        self.escaper_spd = np.zeros([2], dtype=np.float32)
        self._init_pos()

    def _init_pos(self):
        # the boundary
        x_min = SCREEN_WHIDTH / 3
        x_max = 2 * SCREEN_WHIDTH / 3
        y_min = SCREEN_HEIGHT / 3
        y_max = 2 * SCREEN_HEIGHT / 3
        self.escaper_pos = np.array([uniform(x_min + self.collide_min, x_max - self.collide_min),
                                     uniform(y_min + self.collide_min, y_max - self.collide_min)], dtype=np.float32)
        self.hunter_pos = np.zeros([4, 2], dtype=np.float32)
        self.hunter_pos[0] = [uniform(0, x_min - self.collide_min), uniform(0, y_min - self.collide_min)]
        self.hunter_pos[1] = [uniform(x_max + self.collide_min, SCREEN_WHIDTH), uniform(0, y_min - self.collide_min)]
        self.hunter_pos[2] = [uniform(0, x_min - self.collide_min), uniform(y_max + self.collide_min, SCREEN_HEIGHT)]
        self.hunter_pos[3] = [uniform(x_max + self.collide_min, SCREEN_WHIDTH),
                              uniform(y_max + self.collide_min, SCREEN_HEIGHT)]

    def frame_step(self, input_actions):
        # update the pos and speed
        self.move(input_actions)

        # update the display
        screen.fill(background)
        for i in range(len(self.hunter_pos)):
            pygame.draw.rect(screen, hunter_color[i],
                             ((self.hunter_pos[i][0] - self.hunter_radius,
                                   self.hunter_pos[i][1] - self.hunter_radius),
                                  (self.hunter_radius*2, self.hunter_radius*2)))
        pygame.draw.rect(screen, escaper_color,
                           ((self.escaper_pos[0] - self.escaper_radius, self.escaper_pos[1] - self.escaper_radius),
                                (self.escaper_radius*2,  self.escaper_radius*2)))
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        robot_state = [self.escaper_pos[0], self.hunter_pos[0][0], self.hunter_pos[1][0], self.hunter_pos[2][0],self.hunter_pos[3][0],
                       self.escaper_pos[1], self.hunter_pos[0][1], self.hunter_pos[1][1], self.hunter_pos[2][1],self.hunter_pos[3][1],
                       self.escaper_spd[0], self.hunter_spd[0][0], self.hunter_spd[1][0], self.hunter_spd[2][0],self.hunter_spd[3][0],
                       self.escaper_spd[1], self.hunter_spd[0][1], self.hunter_spd[1][1], self.hunter_spd[2][1],self.hunter_spd[3][1],]
        return image_data, robot_state


    def move(self, input_actions):
        robot_n = len(input_actions)
        for i in range(robot_n - 1):#hunters
            if input_actions[i] == 1:  # up, update y_speed
                self.hunter_spd[i][1] -= self.hunter_acc * self.delta_t
            elif input_actions[i] == 2:  # down
                self.hunter_spd[i][1] += self.hunter_acc * self.delta_t
            elif input_actions[i] == 3:  # left, update x_speed
                self.hunter_spd[i][0] -= self.hunter_acc * self.delta_t
            elif input_actions[i] == 4:  # right
                self.hunter_spd[i][0] += self.hunter_acc * self.delta_t
            else:
                pass

            if self.hunter_spd[i][0] < -self.hunter_spd_max:
                self.hunter_spd[i][0] = -self.hunter_spd_max
            elif self.hunter_spd[i][0] > self.hunter_spd_max:
                self.hunter_spd[i][0] = self.hunter_spd_max

            if self.hunter_spd[i][1] < -self.hunter_spd_max:
                self.hunter_spd[i][1] = -self.hunter_spd_max
            elif self.hunter_spd[i][1] > self.hunter_spd_max:
                self.hunter_spd[i][1] = self.hunter_spd_max
            else:
                pass

            self.hunter_pos[i] += self.hunter_spd[i] * self.delta_t
            if self.hunter_pos[i][0] < 0:
                self.hunter_pos[i][0] = 0
                self.hunter_spd[i][0] = 0
            elif self.hunter_pos[i][0] > SCREEN_WHIDTH:
                self.hunter_pos[i][0] = SCREEN_WHIDTH
                self.hunter_spd[i][0] = 0
            else:
                pass

            if self.hunter_pos[i][1] < 0:
                self.hunter_pos[i][1] = 0
                self.hunter_spd[i][1] = 0
            elif self.hunter_pos[i][1] > SCREEN_HEIGHT:
                self.hunter_pos[i][1] = SCREEN_HEIGHT
                self.hunter_spd[i][1] = 0

        #escaper
        if input_actions[robot_n - 1] == 1:  # up, update y_speed
            self.escaper_spd[1] -= self.escaper_acc * self.delta_t
        elif input_actions[robot_n - 1] == 2:  # down
            self.escaper_spd[1] += self.escaper_acc * self.delta_t
        elif input_actions[robot_n - 1] == 3:  # left, update x_speed
            self.escaper_spd[0] -= self.escaper_acc * self.delta_t
        elif input_actions[robot_n - 1] == 4:  # right
            self.escaper_spd[0] += self.escaper_acc * self.delta_t
        else:
            pass

        if self.escaper_spd[0] < -self.escaper_spd_max:
            self.escaper_spd[0] = -self.escaper_spd_max
        elif self.escaper_spd[0] > self.escaper_spd_max:
            self.escaper_spd[0] = self.escaper_spd_max
        else:
            pass

        if self.escaper_spd[1] < -self.escaper_spd_max:
            self.escaper_spd[1] = -self.escaper_spd_max
        elif self.escaper_spd[1] > self.escaper_spd_max:
            self.escaper_spd[1] = self.escaper_spd_max
        else:
            pass
        self.escaper_pos += self.escaper_spd * self.delta_t
