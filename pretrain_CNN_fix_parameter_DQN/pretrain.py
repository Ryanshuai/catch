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

    def add(self, fi, robot_state):
        flattened_fi = np.reshape(fi, [28224])
        experience = np.reshape(np.array([flattened_fi, robot_state]),[1,5])
        if len(self.memory) + len(experience) >= self.memory_size:
            self.memory[0:(len(experience) + len(self.memory)) - self.memory_size] = []
        self.memory.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.memory, size)), [size, 5])

def next_step(action):

    nextObservation = np.zeros(shape=[84, 84, 4], dtype = np.uint8)
    for i in range(4):

        next_image, robot_state = env.frame_step(action)

        next_image = cv2.cvtColor(cv2.resize(next_image, (84, 84)), cv2.COLOR_BGR2GRAY)
        nextObservation[:, :, i] = next_image
    return nextObservation, robot_state


action_num = 5
robot_num = 5
memory = Position_Speed_Memory()

env = ENV()


for iii in range(100000):
    action = np.random.randint(0, action_num, size=robot_num)
    nextObservation, robot_state = next_step(action)
    memory.add(nextObservation, robot_state)


total_train_step = 1000000

pretrain_net = HA.PretrainQNetwork()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

with tf.Session(config=tf_config) as sess:
    tf_writer = tf.summary.FileWriter('logs/')
    tf_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    for train_step in range(total_train_step):

        trainBatch = memory.sample(32)
        # Below we perform the Double-DQN update to the target Q-values

        #Update the network with our target values.
        if(train_step%100==0):
            _, tf_summary = sess.run([pretrain_net.optimize,pretrain_net.merged_summary],
                                    feed_dict={pretrain_net.flattened_batch_fi: np.vstack(trainBatch[:, 0]),
                                                pretrain_net.robots_state: })
            tf_writer.add_summary(tf_summary, train_step)
        else:
            _ = sess.run([pretrain_net.optimize],
                          feed_dict={pretrain_net.flattened_batch_fi: np.vstack(trainBatch[:, 0]),
                                     pretrain_net.robots_state:})
