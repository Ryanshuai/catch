
import cv2
from environment import ENV
from escaper_agent import Escaper_Agent
from hunter_agent import Hunter_Agent
import numpy as np

EPISODE_NUMS = 1000000
img_w = 84
img_h = 84
ACTION_UPDATE_FREQUENCY = 4
START_LEARN = 5000

def next_step(action):
    nextObservation = np.zeros(shape=[img_w, img_w, 4], dtype = np.uint8)
    reward_h_sum = 0
    reward_e_sum = 0
    terminal = False
    for i in range(ACTION_UPDATE_FREQUENCY):
        next_image, reward_h, reward_e, terminal = env.frame_step(action)
        reward_h_sum += reward_h
        reward_e_sum += reward_e
        # terminal = True, flappyBird is inited automatically
        if terminal:
            break
        next_image = cv2.cvtColor(cv2.resize(next_image, (img_w, img_w)), cv2.COLOR_BGR2GRAY)
        nextObservation[:, :,i] = next_image
    return nextObservation, reward_h_sum, reward_e_sum, terminal


def hunting():
    for episode in range(EPISODE_NUMS):
        init_action = np.zeros([5],dtype = np.float32)  # input_actions[1] == 1: flap the bird
        observation, reward_h, reward_e, terminal = next_step(init_action)
        hunter_score = np.zeros([4])
        escaper_score = 0
        while not terminal:
            action = np.zeros([5], dtype = np.float32)
            action[0:4] = hunter.choose_action(observation)
            action[4] = escaper.choose_action(observation)

            nextObservation, reward_h, reward_e, terminal = next_step(action)
            hunter_score+=reward_h
            escaper_score+=reward_e
            hunter.store_transition(observation, action[0:4], reward_h, nextObservation, terminal)
            escaper.store_transition(observation, action[4], reward_e, nextObservation, terminal)
            hunter_printer = hunter.learn()
            escaper_printer = escaper.learn()
            observation = nextObservation
        print('hunter_score:',hunter_score,'\tescaper_score:',escaper_score)
        print('--------')

if __name__ == '__main__':
    env = ENV()
    escaper = Escaper_Agent()
    hunter = Hunter_Agent()
    hunting()