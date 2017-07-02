"""
This part of code is the environment.
This is environment1:four hunter chase a stationary escaper
Using Tensorflow to build the neural network.
"""

import numpy as np
import tensorflow as tf
import pygame
from random import uniform

FPS = 90
SCREEN_WHIDTH = 672
SCREEN_HEIGHT = 672

#init the game
pygame.init()
FPSCLOCK = pygame.time.Clock()
screen = pygame.display.set_mode([SCREEN_WHIDTH, SCREEN_HEIGHT])
pygame.display.set_caption('hunting')

# load resources
background = (255, 255, 255) #white
hunter_color = ((0, 0, 255),(255, 0, 0),(0, 255, 0),(255, 255, 0)) #black
escaper_color = (0, 0, 0) #red


class ENV:
    def __init__(self):
        self.hunter_sidelen = 8
        self.escaper_sidelen = 8
        self.max_pos = np.array([SCREEN_WHIDTH, SCREEN_HEIGHT])
        # self.catch_angle_max = np.pi*3/4
        self.catch_dis = 30.
        self.collide_min = self.hunter_sidelen + self.escaper_sidelen + 2.
        # the center pos, x : [0, SCREEN_WHIDTH], y: [0, SCREEN_HEIGHT]
        self.delta_t = 0.1 # 100ms
        self.hunter_acc = 200
        self.escaper_acc = 0
        self.hunter_spd_max = 100 # 5 pixels once
        self.escaper_spd_max = 0
        self.hunter_spd = np.zeros([4,2],dtype=np.float32)
        self.escaper_spd = np.zeros([2],dtype=np.float32)
        self._init_pos()

    def frame_step(self, input_actions):
        # input_actions: 0->stay, 1->up, 2->down, 3->left, 4->right
        terminal = False
        reward_hunter = np.zeros([4],np.float32)

        #update the pos and speed
        self.move(input_actions)

        # check is_collide
        reward_hunter += self.is_collide() # only collide hunters have rewards

        # check is_cached or is_escaped
        if self.is_catched():
            norm_reletive_dis = [np.linalg.norm(i - self.escaper_pos) for i in self.hunter_pos]
            reward_hunter += [int(i<self.catch_dis) for i in  norm_reletive_dis]
            self.__init__()
            terminal = True

        # update the display
        screen.fill(background)
        for i in range(len(self.hunter_pos)):
            pygame.draw.rect(screen, hunter_color[i],
                             ((self.hunter_pos[i][0] - self.hunter_sidelen,
                               self.hunter_pos[i][1] - self.hunter_sidelen),
                              (self.hunter_sidelen * 2, self.hunter_sidelen * 2)))
        pygame.draw.rect(screen, escaper_color,
                         ((self.escaper_pos[0] - self.escaper_sidelen, self.escaper_pos[1] - self.escaper_sidelen),
                          (self.escaper_sidelen * 2, self.escaper_sidelen * 2)))
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward_hunter, terminal


    def _init_pos(self):
        # the boundary
        x_min = SCREEN_WHIDTH/3
        x_max = 2*SCREEN_WHIDTH/3
        y_min = SCREEN_HEIGHT/3
        y_max = 2*SCREEN_HEIGHT/3
        self.escaper_pos = np.array([uniform(x_min+self.collide_min, x_max-self.collide_min),
                                     uniform(y_min+self.collide_min, y_max-self.collide_min)], dtype=np.float32)
        self.hunter_pos = np.zeros([4,2], dtype=np.float32)
        self.hunter_pos[0] = [uniform(0, x_min-self.collide_min), uniform(0, y_min- self.collide_min)]
        self.hunter_pos[1] = [uniform(x_max+self.collide_min, SCREEN_WHIDTH), uniform(0, y_min- self.collide_min)]
        self.hunter_pos[2] = [uniform(0, x_min-self.collide_min), uniform(y_max+self.collide_min, SCREEN_HEIGHT)]
        self.hunter_pos[3] = [uniform(x_max+self.collide_min, SCREEN_WHIDTH), uniform(y_max + self.collide_min, SCREEN_HEIGHT)]


    def is_catched(self):
        norm_reletive_dis = [np.linalg.norm(i - self.escaper_pos) for i in self.hunter_pos]
        if any([i<self.catch_dis for i in  norm_reletive_dis]):
            return True
        return False


    def is_collide(self):
        reward_collide = np.zeros([4], np.float32)
        radio = -0.01
        for i in range(4):
            if (self.hunter_pos[i][0]<0 or self.hunter_pos[i][0]>SCREEN_WHIDTH or
                        self.hunter_pos[i][1]<0 or self.hunter_pos[i][1]>SCREEN_HEIGHT):
                reward_collide[i] = radio

        for i in range(3):
            for j in range(i+1, 4):
               if( np.linalg.norm([self.hunter_pos[i][0]-self.hunter_pos[j][0],
                                self.hunter_pos[i][1] - self.hunter_pos[j][1]]) <= self.collide_min):
                    reward_collide[i] += radio
                    reward_collide[j] += radio
        return reward_collide


    def move(self, input_actions):
        length = len(input_actions)
        for i in range(length):
            if input_actions[i] == 1: #up, update y_speed
                self.hunter_spd[i][1] -= self.hunter_acc * self.delta_t
            elif input_actions[i] == 2: #down
                self.hunter_spd[i][1] += self.hunter_acc * self.delta_t
            elif input_actions[i] == 3: #left, update x_speed
                self.hunter_spd[i][0] -= self.hunter_acc * self.delta_t
            elif input_actions[i] == 4: #right
                self.hunter_spd[i][0] += self.hunter_acc * self.delta_t

            if self.hunter_spd[i][0] < -self.hunter_spd_max:
                self.hunter_spd[i][0] = -self.hunter_spd_max
            elif self.hunter_spd[i][0] > self.hunter_spd_max:
                self.hunter_spd[i][0] = self.hunter_spd_max

            if self.hunter_spd[i][1] < -self.hunter_spd_max:
                self.hunter_spd[i][1] = -self.hunter_spd_max
            elif self.hunter_spd[i][1] > self.hunter_spd_max:
                self.hunter_spd[i][1] = self.hunter_spd_max

            self.hunter_pos[i] += self.hunter_spd[i] * self.delta_t
            if self.hunter_pos[i][0] < 0:
                self.hunter_pos[i][0] = 0
                self.hunter_spd[i][0] = 0
            elif self.hunter_pos[i][0] > SCREEN_WHIDTH:
                self.hunter_pos[i][0] = SCREEN_WHIDTH
                self.hunter_spd[i][0] = 0

            if self.hunter_pos[i][1] < 0:
                self.hunter_pos[i][1] = 0
                self.hunter_spd[i][1] = 0
            elif self.hunter_pos[i][1] > SCREEN_HEIGHT:
                self.hunter_pos[i][1] = SCREEN_HEIGHT
                self.hunter_spd[i][1] = 0