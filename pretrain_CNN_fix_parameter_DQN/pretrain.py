import hunter_agent as HA
import tensorflow as tf
from collections import Counter
import numpy as np
import cv2
from pretrain_env import ENV
import random

class Position_Speed_Memory():
    def __init__(self, memory_size=100000):
        self.memory = []
        self.memory_size = memory_size

    def add(self, fi, hunter_pos, hunter_spd, escaper_pos, escaper_spd):
        flattened_fi = np.reshape(fi, [28224])
        experience = np.reshape(np.array([flattened_fi, hunter_pos, hunter_spd, escaper_pos, escaper_spd]),[1,5])
        if len(self.memory) + len(experience) >= self.memory_size:
            self.memory[0:(len(experience) + len(self.memory)) - self.memory_size] = []
        self.memory.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.memory, size)), [size, 5])

def next_step(action):

    nextObservation = np.zeros(shape=[84, 84, 4], dtype = np.uint8)
    for i in range(4):

        next_image, hunter_pos, hunter_spd, escaper_pos, escaper_spd = env.frame_step(action)

        next_image = cv2.cvtColor(cv2.resize(next_image, (84, 84)), cv2.COLOR_BGR2GRAY)
        nextObservation[:, :, i] = next_image
    return nextObservation, hunter_pos, hunter_spd, escaper_pos, escaper_spd


action_num = 5
robot_num = 5
memory = Position_Speed_Memory()

env = ENV()


for iii in range(100000):
    action = np.random.randint(0, action_num, size=robot_num)
    nextObservation, hunter_pos, hunter_spd, escaper_pos, escaper_spd = next_step(action)
    memory.add(nextObservation, hunter_pos, hunter_spd, escaper_pos, escaper_spd)


total_train_step = 1000000

pretrain_net = HA.PretrainQNetwork()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    for train_step in range(total_train_step):

        trainBatch = memory.sample(32)
        # Below we perform the Double-DQN update to the target Q-values
        spd0, pos0, spd1, pos1, spd2, pos2, spd3, pos3, spd4, pos4 = sess.run(pretrain_net.h_fc6, feed_dict={pretrain_net.flattened_batch_fi: np.vstack(trainBatch[:, 3])})#[bs,act_num]
        escaper_pos = pos0
        escaper_spd = spd0
        #Update the network with our target values.
        _, loss = sess.run(
            [self.trn_net.updateModel, self.trn_net.loss],
            feed_dict={self.trn_net.flattened_batch_fi: np.vstack(trainBatch[:, 0]),
                       self.trn_net.targetQ: targetQ,
                       self.trn_net.actions: trainBatch[:, 1]})
        return loss

