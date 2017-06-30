
import cv2
from environment import ENV
from escaper_agent import Escaper_Agent
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
        next_image = ys_resize(next_image)
        nextObservation[:, :,i] = next_image
    return nextObservation, reward_h_sum, reward_e_sum, terminal


def hunting():
    for episode in range(EPISODE_NUMS):
        init_action = np.zeros([5],dtype = np.float32)  # input_actions[1] == 1: flap the bird
        observation, reward_h, reward_e, terminal = next_step(init_action)
        hunter_score = np.zeros([4])
        escaper_score = 0
        escaper_printer = [0,0,0,0]
        hunter_printer = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
        while not terminal:
            action = np.zeros([5], dtype = np.float32)
            for i in range(4):
                action[i] = hunter[i].choose_action(observation)
            action[4] = escaper.choose_action(observation)
            nextObservation, reward_h, reward_e, terminal = next_step(action)
            hunter_score+=reward_h
            escaper_score+=reward_e
            for i in range(4):
                hunter[i].store_transition(observation, action[i], reward_h[i], nextObservation, terminal)
            escaper.store_transition(observation, action[4], reward_e, nextObservation, terminal)
            for i in range(4):
                hunter_printer[i] = hunter[i].learn()
            escaper_printer = escaper.learn()
            observation = nextObservation
        print('exploration',hunter_printer[0][1],'train_step',hunter_printer[0][2],'update',hunter_printer[0][3])
        print('h0_loss',hunter_printer[0][4],'h1_loss',hunter_printer[1][4],'h2_loss',hunter_printer[2][4],'h3_loss',hunter_printer[3][4])
        print('hunter0_action_value',hunter_printer[0][0],'h1av',hunter_printer[1][0],'h2av',hunter_printer[2][0],'h3av',hunter_printer[3][0])
        print('hunter_score:',hunter_score,'\tescaper_score:',escaper_score)
        print('----------------------------------------------------------------------------------------------escaper_loss',escaper_printer[4])

load_mode0 = 'Hunter/mode1/mode0'
load_mode1 = 'Hunter/mode2/mode0'
load_mode2 = 'Hunter/mode3/mode0'
load_mode3 = 'Hunter/mode4/mode0'
save_mode0 = 'Hunter/model1/model1/model.ckpt'
save_mode1 = 'Hunter/model2/model1/model.ckpt'
save_mode2 = 'Hunter/model3/model1/model.ckpt'
save_mode3 = 'Hunter/model4/model1/model.ckpt'
frozen_net_collecion_name0 = 'hunter0_frozen_net'
frozen_net_collecion_name1 = 'hunter1_frozen_net'
frozen_net_collecion_name2 = 'hunter2_frozen_net'
frozen_net_collecion_name3 = 'hunter3_frozen_net'
training_net_collecion_name0 = 'hunter0_training_net'
training_net_collecion_name1 = 'hunter1_training_net'
training_net_collecion_name2 = 'hunter2_training_net'
training_net_collecion_name3 = 'hunter3_training_net'

if __name__ == '__main__':
    env = ENV()
    escaper = Escaper_Agent()
    hunter = [Hunter_Agent(load_mode0, save_mode0, frozen_net_collecion_name0, training_net_collecion_name0),
              Hunter_Agent(load_mode1, save_mode1, frozen_net_collecion_name1, training_net_collecion_name1),
              Hunter_Agent(load_mode2, save_mode2, frozen_net_collecion_name2, training_net_collecion_name2),
              Hunter_Agent(load_mode3, save_mode3, frozen_net_collecion_name3, training_net_collecion_name3)]
    hunting()