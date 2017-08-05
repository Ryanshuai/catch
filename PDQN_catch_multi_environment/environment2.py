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
background = (255, 255, 255)
hunter_color = (0, 0, 255)
escaper_color = (0, 0, 0)


class ENV:
    def __init__(self):

        self.hunter_radius = 8
        self.escaper_radius = 8
        self.max_pos = np.array([SCREEN_WHIDTH, SCREEN_HEIGHT])
        self.catch_dis = 50.
        self.delta_t = 0.1 # 100ms
        self.hunter_acc = 200
        self.escaper_acc = 200
        self.hunter_spd_max = 100 # 5 pixels once
        self.escaper_spd_max = 50
        self.hunter_spd = np.zeros([2],dtype=np.float32)
        self.escaper_spd = np.zeros([2],dtype=np.float32)
        self._init_pos()

    def _init_pos(self):
        # the boundary
        x_1_3rd = SCREEN_WHIDTH/3
        x_2_3rd = 2*SCREEN_WHIDTH/3
        y_1_3rd = SCREEN_HEIGHT/3
        y_2_3rd = 2*SCREEN_HEIGHT/3
        self.escaper_pos = np.array([uniform(x_1_3rd+20, x_2_3rd-20),
                                     uniform(y_1_3rd+20, y_2_3rd-20)], dtype=np.float32)
        randpos = random.randint(0, 3)
        if randpos == 0:
            self.hunter_pos = [uniform(0, x_1_3rd), uniform(0, y_1_3rd)]
        elif randpos == 1:
            self.hunter_pos = [uniform(x_2_3rd, SCREEN_WHIDTH), uniform(0, y_1_3rd)]
        elif randpos == 2:
            self.hunter_pos = [uniform(0, x_1_3rd), uniform(y_2_3rd, SCREEN_HEIGHT)]
        else:
            self.hunter_pos = [uniform(x_2_3rd, SCREEN_WHIDTH), uniform(y_2_3rd, SCREEN_HEIGHT)]


    def frame_step(self, input_actions):
        # input_actions: 0->stay, 1->up, 2->down, 3->left, 4->right
        terminal = False
        reward_hunter = 0
        reward_escaper = 0

        #update the pos and speed
        self.move(input_actions)

        # check is_cached or is_escaped
        if self.is_catched():
            reward_hunter = 1
            reward_escaper = -1
            self.__init__()
            terminal = True
        elif self.is_escaped():
            reward_hunter = 0
            reward_escaper = 1
            self.__init__()
            terminal = True

        # update the display
        screen.fill(background)
        pygame.draw.rect(screen, hunter_color,
                             ((self.hunter_pos[0] - self.hunter_radius,self.hunter_pos[1] - self.hunter_radius),
                              (self.hunter_radius * 2, self.hunter_radius * 2)))
        pygame.draw.rect(screen, escaper_color,
                         ((self.escaper_pos[0] - self.escaper_radius, self.escaper_pos[1] - self.escaper_radius),
                          (self.escaper_radius * 2, self.escaper_radius * 2)))
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward_hunter, reward_escaper, terminal


    def is_catched(self):
        norm_relative_dis = [np.linalg.norm(self.hunter_pos - self.escaper_pos)]
        if norm_relative_dis<self.catch_dis:
            return True
        return False


    def is_escaped(self):
        if any([i>0 for i in self.escaper_pos-self.max_pos]+[i<0 for i in self.escaper_pos]):
            return True
        return False


    def move(self, input_actions):
        length = len(input_actions)
        for i in range(length -1):
            if input_actions[i] == 1: #up, update y_speed
                self.hunter_spd[1] -= self.hunter_acc * self.delta_t
            elif input_actions[i] == 2: #down
                self.hunter_spd[1] += self.hunter_acc * self.delta_t
            elif input_actions[i] == 3: #left, update x_speed
                self.hunter_spd[0] -= self.hunter_acc * self.delta_t
            elif input_actions[i] == 4: #right
                self.hunter_spd[0] += self.hunter_acc * self.delta_t

            if self.hunter_spd[0] < -self.hunter_spd_max:
                self.hunter_spd[0] = -self.hunter_spd_max
            elif self.hunter_spd[0] > self.hunter_spd_max:
                self.hunter_spd[0] = self.hunter_spd_max

            if self.hunter_spd[1] < -self.hunter_spd_max:
                self.hunter_spd[1] = -self.hunter_spd_max
            elif self.hunter_spd[1] > self.hunter_spd_max:
                self.hunter_spd[1] = self.hunter_spd_max

            self.hunter_pos += self.hunter_spd * self.delta_t
            if self.hunter_pos[0] < 0:
                self.hunter_pos[0] = 0
                self.hunter_spd[0] = 0
            elif self.hunter_pos[0] > SCREEN_WHIDTH:
                self.hunter_pos[0] = SCREEN_WHIDTH
                self.hunter_spd[0] = 0

            if self.hunter_pos[1] < 0:
                self.hunter_pos[1] = 0
                self.hunter_spd[1] = 0
            elif self.hunter_pos[1] > SCREEN_HEIGHT:
                self.hunter_pos[1] = SCREEN_HEIGHT
                self.hunter_spd[1] = 0

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