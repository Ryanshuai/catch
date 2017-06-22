"""
This part of code is the environment.
Using Tensorflow to build the neural network.
"""

import numpy as np
import tensorflow as tf
import pygame


class ENV:
    def __init__(self):
        self.fps = 90
        self.screen_height = 500
        self.screen_width = 500

        self.max_pos = [500.,500.]
        self.escaper_pos = np.zeros([2])
        self.hunter_pos = np.zeros([4,2])
        self.catch_dis = 4.

    def frame_step(self, input_actions):
        pass


    def is_terminal(self):
        if self.is_catched() or self.is_escaped():
            return True
        return False

    def is_catched(self):
        reletive_dis = [i - self.escaper_pos for i in self.hunter_pos]
        reletive_dis = list(filter(lambda x:np.linalg.norm(x)<self.catch_dis, reletive_dis))
        if len(reletive_dis)<3:
            return False
        x = [i[0] for i in reletive_dis]
        y = [i[1] for i in reletive_dis]
        angle_e_h = np.arctan2(y, x)
        angle_e_h.sort()
        error_angle_e_h = angle_e_h[list(range(1,len(angle_e_h)))+[0]] - angle_e_h
        error_angle_e_h[len(error_angle_e_h) - 1] += 2 * np.pi
        if any([abs(i)>np.pi*3/4 for i in error_angle_e_h]):
            return False
        return True


    def is_escaped(self):
        if any([i>0 for i in self.escaper_pos-self.max_pos]+[i<0 for i in self.escaper_pos]):
            return True
        return False