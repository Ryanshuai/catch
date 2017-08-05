
import cv2
from environment_1 import ENV
from hunter_agent import Hunter_Agent
import numpy as np
from Ryan_resize import ys_resize

EPISODE_NUMS = 1000000
img_w = 84
img_h = 84
ACTION_UPDATE_FREQUENCY = 4
START_LEARN = 5000

def next_step(action):
    nextObservation = np.zeros(shape=[img_w, img_w, 4], dtype = np.uint8)
    reward_h_sum = 0
    terminal = False
    for i in range(ACTION_UPDATE_FREQUENCY):
        next_image, reward_h, terminal = env.frame_step(action)
        reward_h_sum += reward_h

        if terminal:
            break
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
        next_image = ys_resize(next_image)
        nextObservation[:, :,i] = next_image
    return nextObservation, reward_h_sum, terminal


def hunting():
    for episode in range(EPISODE_NUMS):
        init_action = np.zeros([4],dtype = np.float32)  # input_actions[1] == 1: flap the bird
        observation, reward_h, terminal = next_step(init_action)
        hunter_score = np.zeros([4])
        hunter_printer = [[0],[0],[0],[0]]
        while not terminal:
            action = hunter.choose_action(observation)
            nextObservation, reward_h, terminal = next_step(action[0])
            hunter_score+=reward_h
            hunter.store_transition(observation, action, reward_h, nextObservation, terminal)
            hunter_printer = hunter.learn()
            observation = nextObservation
        print('exploration',hunter_printer[1],'train_step',hunter_printer[2],'update',hunter_printer[3],'hunter_loss',hunter_printer[4])
        print('hunter0_action_value',hunter_printer[0][0])


if __name__ == '__main__':
    env = ENV()
    hunter = Hunter_Agent()
    hunting()