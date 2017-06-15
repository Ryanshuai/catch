"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class Escaper_Agent:
    def __init__(self,
                 n_actions,
                 observation_w = 84,
                 observation_h = 84,
                 agent_history_length = 4,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 target_network_update_frequency=10000,
                 memory_size=50000, # 2.6G
                 batch_size=32,
                 output_graph=False,
                 ):

        self.n_actions = n_actions #up down left right
        self.w = observation_w
        self.h = observation_h
        self.m = agent_history_length # w*h*m, this is the parameter of m

        self.batch_size = batch_size
        self.memory_size = memory_size  # replay memory size
        self.target_update_frequency = target_network_update_frequency #target network update frequency
        self.gamma = reward_decay
        self.lr = learning_rate
        self.initial_exploration = 1
        self.final_exploration = 0.1
        self.epsilon = self.initial_exploration

        self.memory = {'s': np.ones(shape=[memory_size, self.w, self.h, self.m], dtype=np.uint8),#0-255
                  'a': np.ones(shape=[memory_size, ], dtype=np.int8),
                  'r': np.ones(shape=[memory_size, ], dtype=np.int8),
                  's_': np.ones(shape=[memory_size, self.w, self.h, self.m], dtype=np.uint8)}

        self.memory_counter = 0
        self.learn_step_counter = 0
        # self.learning_rate = 0.00025
        # self.gredient_momentum = 0.95
        # self.squared_gredient_momentum = 0.95
        # self.min_squared_gradient = 0.01


        self._build_net()# consist of [target_net, evaluate_net]
        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        def build_layers(s, c_names, keep_prob):
            ## conv1 layer ##
            W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 4, 32], stddev=0.1), collections=c_names)
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]),collections=c_names)
            conv1 = tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding='SAME')
            h_conv1 = tf.nn.relu(conv1 + b_conv1)

            ## conv2 layer ##
            W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1), collections=c_names)
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]),collections=c_names)
            conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME')
            h_conv2 = tf.nn.relu(conv2 + b_conv2)

            ## conv3 layer ##
            W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1),collections=c_names)
            b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]),collections=c_names)
            conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            h_conv3 = tf.nn.relu(conv3 + b_conv3)

            # [n_samples, 11, 11, 64] ->> [n_samples, 7744]
            h_pool3_flat = tf.reshape(h_conv3, [-1, 7744])

            ## fc4 layer ##
            W_fc4 = tf.Variable(tf.truncated_normal([7744, 4096], stddev=0.1),collections=c_names)
            b_fc4 = tf.Variable(tf.constant(0.1, shape=[4096]),collections=c_names)
            h_fc4 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc4) + b_fc4)
            h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

            ## fc5 layer ##
            W_fc5 = tf.Variable(tf.truncated_normal([4096, self.n_actions], stddev=0.1),collections=c_names)
            b_fc5 = tf.Variable(tf.constant(0.1, shape=[self.n_actions]),collections=c_names)
            h_fc5 = tf.matmul(h_fc4_drop, W_fc5) + b_fc5
            return h_fc5

        # all inputs
        self.s = tf.placeholder(tf.float32,
                shape=[None, self.w, self.h, self.m],
                                name = 's') / 255
        self.s_ = tf.placeholder(tf.float32,
                shape=[None, self.w, self.h, self.m],
                                 name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.keep_prob = tf.placeholder(tf.float32)

        # ------------------ build evaluate_net ------------------
        c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        self.q_eval = build_layers(self.s, c_names, self.keep_prob)
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        self.q_next = build_layers(self.s_, c_names, self.keep_prob)
        self.q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1)

    def store_transition(self, s, a, r, s_):
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory['s'][index] = s
        self.memory['a'][index] = a
        self.memory['r'][index] = r
        self.memory['s_'][index] = s_

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        #[84,84,3] - > [1,84,84,3]
        observation = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        # check to reeplace target parameters
        if self.learn_step_counter % self.target_update_frequency == 0:
            self._replace_target_params()
            print('target_params_replaced')
            if self.epsilon < self.final_exploration:
                self.epsilon += 0.009
            else:
                self.epsilon = self.final_exploration


        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_s = self.memory['s'][sample_index]
        batch_a = self.memory['a'][sample_index]
        batch_r = self.memory['r'][sample_index]
        batch_s_ = self.memory['s_'][sample_index]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_s_,  # fixed params
                self.s: batch_s,  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_a.astype(int)
        reward = batch_r

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_s,
                                                self.q_target: q_target})