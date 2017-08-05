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
        self.hunter_radius = 8
        self.escaper_radius = 8
        self.max_pos = np.array([SCREEN_WHIDTH, SCREEN_HEIGHT])
        self.catch_dis = 50.
        self.collide_min = self.hunter_radius + self.escaper_radius + 2.
        self.delta_t = 0.1 # 100ms
        self.hunter_acc = 200
        self.escaper_acc = 200
        self.hunter_spd_max = 100 # 5 pixels once
        self.escaper_spd_max = 30
        self.hunter_spd = np.zeros([4,2],dtype=np.float32)
        self.escaper_spd = np.zeros([2],dtype=np.float32)
        self._init_pos()

    def _init_pos(self):
        x_1_3rd = SCREEN_WHIDTH/3
        x_2_3rd = 2*SCREEN_WHIDTH/3
        y_1_3rd = SCREEN_HEIGHT/3
        y_2_3rd = 2*SCREEN_HEIGHT/3
        self.escaper_pos = np.array([uniform(x_1_3rd+self.collide_min, x_2_3rd-self.collide_min),
                                     uniform(y_1_3rd+self.collide_min, y_2_3rd-self.collide_min)], dtype=np.float32)
        self.hunter_pos = np.zeros([4,2], dtype=np.float32)
        self.hunter_pos[0] = [uniform(0, x_1_3rd-self.collide_min), uniform(0, y_1_3rd- self.collide_min)]
        self.hunter_pos[1] = [uniform(x_2_3rd+self.collide_min, SCREEN_WHIDTH), uniform(0, y_1_3rd- self.collide_min)]
        self.hunter_pos[2] = [uniform(0, x_1_3rd-self.collide_min), uniform(y_2_3rd+self.collide_min, SCREEN_HEIGHT)]
        self.hunter_pos[3] = [uniform(x_2_3rd+self.collide_min, SCREEN_WHIDTH), uniform(y_2_3rd + self.collide_min, SCREEN_HEIGHT)]


    def frame_step(self, input_actions):
        # input_actions: 0->stay, 1->up, 2->down, 3->left, 4->right
        terminal = False
        reward_hunter = np.zeros([4],np.float32)
        reward_escaper = 0

        #update the pos and speed
        self.move(input_actions)

        # check is_collide
        reward_hunter += self.is_collide() # only collide hunters have rewards

        # check is_cached or is_escaped
        if self.is_catched():
            all_reward = [1,1,1,1]
            reward_hunter += all_reward
            reward_escaper = -1
            self.__init__()
            terminal = True
        elif self.is_escaped():
            all_reward = [0,0,0,0]
            reward_hunter += all_reward
            reward_escaper = 1
            self.__init__()
            terminal = True

        # update the display
        screen.fill(background)
        for i in range(len(self.hunter_pos)):
            pygame.draw.rect(screen, hunter_color[i],
                             ((self.hunter_pos[i][0] - self.hunter_radius,self.hunter_pos[i][1] - self.hunter_radius),
                              (self.hunter_radius * 2, self.hunter_radius * 2)))
        pygame.draw.rect(screen, escaper_color,
                         ((self.escaper_pos[0] - self.escaper_radius, self.escaper_pos[1] - self.escaper_radius),
                          (self.escaper_radius * 2, self.escaper_radius * 2)))
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward_hunter, reward_escaper, terminal


    def is_catched(self):
        norm_reletive_dis = [np.linalg.norm(i - self.escaper_pos) for i in self.hunter_pos]
        if all([i < self.catch_dis for i in norm_reletive_dis]):
            return True#,hunter_reward
        return False


    def is_escaped(self):
        if any([i>0 for i in self.escaper_pos-self.max_pos]+[i<0 for i in self.escaper_pos]):
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
        for i in range(length -1):
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

        if input_actions[length -1] == 1:  # up, update y_speed
            self.escaper_spd[1] -= self.escaper_acc * self.delta_t
        elif input_actions[length -1] == 2:  # down
            self.escaper_spd[1] += self.escaper_acc * self.delta_t
        elif input_actions[length -1] == 3:  # left, update x_speed
            self.escaper_spd[0] -= self.escaper_acc * self.delta_t
        elif input_actions[length -1] == 4:  # right
            self.escaper_spd[0] += self.escaper_acc * self.delta_t

        if self.escaper_spd[0] < -self.escaper_spd_max:
            self.escaper_spd[0] = -self.escaper_spd_max
        elif self.escaper_spd[0] > self.escaper_spd_max:
            self.escaper_spd[0] = self.escaper_spd_max

        if self.escaper_spd[1] < -self.escaper_spd_max:
            self.escaper_spd[1] = -self.escaper_spd_max
        elif self.escaper_spd[1] > self.escaper_spd_max:
            self.escaper_spd[1] = self.escaper_spd_max
        self.escaper_pos += self.escaper_spd * self.delta_t