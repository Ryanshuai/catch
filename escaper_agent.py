"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
"""

import numpy as np
import tensorflow as tf

# Deep Q Network off-policy
class Escaper_Agent:
    def __init__(self):
        self.n_actions = 4 #up down left right
        self.n_robot = 1

        self.batch_size = 32
        self.memory_size = 100000  # replay memory size
        self.history_length = 4 #agent history length
        self.target_network_update_frequency = 1000 #target network update frequency
        self.gamma = 0.99  # discount factor
        self.action_repeat = 4
        self.update_frequency = 4
        self.exploration = 1. #initial
        self.final_exploration = 0.1
        self.final_exploration_frame = 100000
        self.replay_start_size = 5000
        self.cost_his = []  # the error of every step
        #used by RMSProp
        self.lr = 0.00025
        self.gredient_momentum = 0.95
        self.squared_gredient_momentum = 0.95
        self.min_squared_gradient = 0.01
        #counter
        self.learn_step_counter = 0  # total learning step
        self.memory_counter = 0
        # w*h*m, this is the parameter of memory
        self.w = 84 #observation_w
        self.h = 84 #observation_h
        self.m = 4 #agent_history_length
        self.memory = {'fi': np.ones(shape=[self.memory_size, self.w, self.h, self.m], dtype=np.uint8),#0-255
                  'a': np.ones(shape=[self.memory_size, ], dtype=np.int8),
                  'r': np.ones(shape=[self.memory_size, ], dtype=np.int8),
                  'fi_': np.ones(shape=[self.memory_size, self.w, self.h, self.m], dtype=np.uint8)}

        self._build_net()# consist of [target_net, evaluate_net]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def _build_net(self):
        def build_layers(s, collection_names, keep_prob):
            ## conv1 layer ##
            W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.1), collections=collection_names)
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), collections=collection_names)
            conv1 = tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding='SAME')
            h_conv1 = tf.nn.relu(conv1 + b_conv1)
            ## conv2 layer ##
            W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1), collections=collection_names)
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), collections=collection_names)
            conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME')
            h_conv2 = tf.nn.relu(conv2 + b_conv2)
            ## conv3 layer ##
            W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), collections=collection_names)
            b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]), collections=collection_names)
            conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            h_conv3 = tf.nn.relu(conv3 + b_conv3)
            # [n_samples, 11, 11, 64] ->> [n_samples, 7744]
            h_pool3_flat = tf.reshape(h_conv3, [-1, 7744])
            ## fc4 layer ##
            W_fc4 = tf.Variable(tf.truncated_normal([7744, 512], stddev=0.1), collections=collection_names)
            b_fc4 = tf.Variable(tf.constant(0.1, shape=[512]), collections=collection_names)
            h_fc4 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc4) + b_fc4)
            h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)
            ## fc5 layer ##
            W_fc5 = tf.Variable(tf.truncated_normal([512, self.n_actions*self.n_robot], stddev=0.1), collections=collection_names)
            b_fc5 = tf.Variable(tf.constant(0.1, shape=[self.n_actions*self.n_robot]), collections=collection_names)
            h_fc5 = tf.matmul(h_fc4_drop, W_fc5) + b_fc5
            return h_fc5

        # all inputs
        self.im_to_evaluate_net = tf.placeholder(tf.float32, shape=[None, self.w, self.h, self.m],name = 'fi') / 255
        self.im_to_target_net = tf.placeholder(tf.float32,shape=[None, self.w, self.h, self.m],name='fi_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.keep_prob = tf.placeholder(tf.float32)

        # ------------------ build evaluate_net ------------------
        col_eval_net = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        self.q_eval = build_layers(self.im_to_evaluate_net, col_eval_net, self.keep_prob)

        # ------------------ build target_net ------------------
        col_targ_net = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        self.q_next = build_layers(self.im_to_target_net, col_targ_net, self.keep_prob)
        self.q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1)

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    def store_transition(self, fi, a, r, fi_):
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory['fi'][index] = fi
        self.memory['a'][index] = a
        self.memory['r'][index] = r
        self.memory['fi_'][index] = fi_
        self.memory_counter += 1


    def choose_action(self, observation):
        observation = observation[np.newaxis, :]#[84,84,4] - > [1,84,84,4]
        if np.random.uniform() < self.exploration: #exploration
            action = np.random.randint(0, self.n_actions)
        else:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.im_to_evaluate_net: observation})
            action = np.argmax(actions_value)
        return action


    def learn(self):
        # check to reeplace target parameters
        if self.learn_step_counter % self.target_network_update_frequency == 0:
            t_params = tf.get_collection('target_net_params')
            e_params = tf.get_collection('eval_net_params')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
            print('target_params_replaced')
            if(self.exploration > self.final_exploration):
                self.exploration -= 0.009
                print('self.exploration changed to',self.exploration)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_fi = self.memory['fi'][sample_index]
        batch_a = self.memory['a'][sample_index]
        batch_r = self.memory['r'][sample_index]
        batch_fi_ = self.memory['fi_'][sample_index]

        q_eval,q_next = self.sess.run(
            [self.q_eval,self.q_next],
            feed_dict={
                self.im_to_evaluate_net: batch_fi, # newest params
                self.im_to_target_net: batch_fi_, # fixed params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_a.astype(int)
        reward = batch_r
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.im_to_evaluate_net: batch_fi,
                                                self.q_target: q_target})

        self.cost_his.append(self.cost)
        self.learn_step_counter += 1


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()