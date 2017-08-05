# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

import cv2
from game import wrapped_flappy_bird as game
from escaper_agent import Escaper_Agent
import numpy as np

EPISODE_NUMS = 1000000
img_w = 84
img_h = 84
ACTION_UPDATE_FREQUENCY = 4
START_LEARN = 5000

def next_step(action):
    nextObservation = np.zeros(shape=[img_w, img_w, ACTION_UPDATE_FREQUENCY], dtype = np.uint8)
    reward = 0
    reward_sum = 0
    terminal = False
    for i in range(ACTION_UPDATE_FREQUENCY):
        next_image, reward, terminal = flappyBird.frame_step(action)
        reward_sum += reward
        # terminal = True, flappyBird is inited automatically
        if terminal:
            break
        next_image = cv2.cvtColor(cv2.resize(next_image, (img_w, img_w)), cv2.COLOR_BGR2GRAY)
        nextObservation[:, :, i] = next_image
    return nextObservation, reward_sum/ACTION_UPDATE_FREQUENCY , terminal


def playFlappyBird():
    for episode in range(EPISODE_NUMS):
        init_action = np.array([1, 0])
        observation, reward, terminal = next_step(init_action)
        while True:
            action_index = brain.choose_action(observation)
            action = np.zeros(shape=[2,])
            action[action_index] = 1
            nextObservation, reward, terminal = next_step(action)
            brain.store_transition(observation, action_index, reward, nextObservation)
            brain.learn()
            observation = nextObservation
            if terminal:
                break

if __name__ == '__main__':
    flappyBird = game.GameState()
    brain = Escaper_Agent()
    playFlappyBird()

