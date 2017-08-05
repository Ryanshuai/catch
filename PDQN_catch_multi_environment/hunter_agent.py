"""
This part of code is the Hunter_Agent.
hunter decisions are made in here.
Using Tensorflow to build the neural network.
"""

import numpy as np
import tensorflow as tf

LOAD_MODEL = 'hunter_model/model0' #load model from here
SAVE_MODEL = 'hunter_model/model0/model.ckpt' #save model to here

class Hunter_Agent:
    def __init__(self):
        self.n_actions = 5 #up down left right remain
        self.n_robot = 4

        self.batch_size = 32
        self.memory_size = 100000  # replay memory size
        self.history_length = 4 #agent history length
        self.frozen_network_update_frequency = 1000 #frozen network update frequency
        self.gamma = 0.99  # discount factor
        self.action_repeat = 4
        self.update_frequency = 4
        self.initial_exploration = 1. #1. #initial
        self.final_exploration = 0.1
        self.exploration = self.initial_exploration 
        self.final_exploration_frame = 100000
        self.replay_start_size = 1000
        #used by RMSProp
        self.lr = 0.00025
        self.min_squared_gradient = 0.01
        #counter and printer
        self.train_step_counter = 1  # total learning step
        self.memory_counter = 1
        self.update_counter = 0
        self.outloss = 0
        self.actions_value = np.zeros([self.n_robot, self.n_actions], dtype=np.float32)
        # w*h*m, this is the parameter of memory
        self.w = 84 #observation_w
        self.h = 84 #observation_h
        self.m = 4 #agent_history_length
        self.memory = {'fi': np.zeros(shape=[self.memory_size, self.w, self.h, self.m], dtype=np.uint8),  # 0-255
                       'a': np.zeros(shape=[self.memory_size, self.n_robot], dtype=np.int8),
                       'r': np.zeros(shape=[self.memory_size, self.n_robot], dtype=np.int8),
                       'Nfi': np.zeros(shape=[self.memory_size, self.w, self.h, self.m], dtype=np.uint8),
                       'done': np.zeros(shape=[self.memory_size, ], dtype=np.uint8)}

        self._build_pnn_net(Train_Times=0) # how times
        self.saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())

        # ------------------ load model ------------------
        ckpt = tf.train.get_checkpoint_state(LOAD_MODEL)
        if ckpt and ckpt.model_checkpoint_path:
            print('loading_model')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)


    def _build_pnn_net(self, Train_Times):
        def build_collum(s, collection_names, h1s=None, h2s=None, h3s=None, h4s=None):
            ## conv1 layer ##
            W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01), collections=collection_names)
            b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]), collections=collection_names)
            conv1 = tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding='SAME')
            h_conv1 = tf.nn.relu(conv1 + b_conv1)

            ## conv2 layer ##
            W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01), collections=collection_names)
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), collections=collection_names)
            conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME')
            if h1s:
                h1s_len = len(h1s)
                U_conv2 = [0]*h1s_len
                uconv2 = [0]*h1s_len
                for i in range(h1s_len):
                    U_conv2[i] = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01), collections=collection_names)
                    uconv2[i] = tf.nn.conv2d(h1s[i], U_conv2[i], strides=[1, 2, 2, 1], padding='SAME')
                h_conv2 = tf.nn.relu(conv2 + b_conv2 + sum(uconv2))
            else:
                h_conv2 = tf.nn.relu(conv2 + b_conv2)
            ## conv3 layer ##
            W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01), collections=collection_names)
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]), collections=collection_names)
            conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            if h2s:
                h2s_len = len(h2s)
                U_conv3 = [0]*h2s_len
                uconv3 = [0]*h2s_len
                for i in range(h2s_len):
                    U_conv3[i] = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01), collections=collection_names)
                    uconv3[i] = tf.nn.conv2d(h2s[i], U_conv3[i], strides=[1, 2, 2, 1], padding='SAME')
                h_conv3 = tf.nn.relu(conv3 + b_conv3 + sum(uconv3))
            else:
                h_conv3 = tf.nn.relu(conv3 + b_conv3)
            # [n_samples, 11, 11, 64] ->> [n_samples, 7744]
            h_conv3_flat = tf.reshape(h_conv3, [-1, 7744])
            ## fc4 layer ##
            W_fc4 = tf.Variable(tf.truncated_normal([7744, 512], stddev=0.01), collections=collection_names)
            b_fc4 = tf.Variable(tf.constant(0.01, shape=[512]), collections=collection_names)
            if h3s:
                h3s_len = len(h3s)
                U_fc4 = [0] * h3s_len
                sum_matnul_h3_U = [0]*h3s_len
                for i in range(h3s_len):
                    U_fc4[i] = tf.Variable(tf.truncated_normal([7744, 512], stddev=0.01), collections=collection_names)
                    sum_matnul_h3_U[i] = tf.matmul(h3s[i], U_fc4[i])
                h_fc4 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc4) + b_fc4 + sum(sum_matnul_h3_U))
            else:
                h_fc4 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc4) + b_fc4)

            ## fc5 layer ##
            W_fc5 = tf.Variable(tf.truncated_normal([512, self.n_actions*self.n_robot], stddev=0.01), collections=collection_names)
            b_fc5 = tf.Variable(tf.constant(0.01, shape=[self.n_actions*self.n_robot]), collections=collection_names)
            if h4s:
                h4s_len = len(h4s)
                U_fc5 = [0]*h4s_len
                sum_matnul_h4_U = [0] * h4s_len
                for i in range(h4s_len):
                    U_fc5[i] = tf.Variable(tf.truncated_normal([512, self.n_actions * self.n_robot], stddev=0.01),collections=collection_names)
                    sum_matnul_h4_U[i] = tf.matmul(h4s[i], U_fc5[i])
                h_fc5 = tf.matmul(h_fc4, W_fc5) + b_fc5 + sum(sum_matnul_h4_U)
            else:
                h_fc5 = tf.matmul(h_fc4, W_fc5) + b_fc5
            h_fc5_reshape = tf.reshape(h_fc5, shape=[-1, self.n_robot, self.n_actions])
            return h_conv1,h_conv2,h_conv3_flat,h_fc4,h_fc5_reshape

        # ------------------ build training_net ---------------------
        self.batch_fi = tf.placeholder(tf.float32, shape=[None, self.w, self.h, self.m]) / 255
        if Train_Times == 0:
            self.training_name = 'col_0_training'
        elif Train_Times == 1:
            self.training_name = 'col_1_training'
        elif Train_Times == 2:
            self.training_name = 'col_2_training'
        col0_training_name = ['col_0_training', tf.GraphKeys.TRAINABLE_VARIABLES]
        col0_train = build_collum(self.batch_fi, col0_training_name)
        col1_training_name = ['col_1_training', tf.GraphKeys.TRAINABLE_VARIABLES]
        col1_train = build_collum(self.batch_fi, col1_training_name)
        col2_training_name = ['col_2_training', tf.GraphKeys.TRAINABLE_VARIABLES]
        col2_train = build_collum(self.batch_fi, col2_training_name, h1s=[col0_train[0], col1_train[0]],
                                  h2s=[col0_train[1], col1_train[1]],
                                  h3s=[col0_train[2], col1_train[2]],
                                  h4s=[col0_train[3], col1_train[3]])
        if Train_Times == 0:
            self.q_fi_from_training_net = col0_train[4]
        elif Train_Times == 1:
            self.q_fi_from_training_net = col1_train[4]
        elif Train_Times == 2:
            self.q_fi_from_training_net = col2_train[4]
        self.batch_a = tf.placeholder(tf.int32, [None, self.n_robot])  # input Action
        a_one_hot = tf.one_hot(self.batch_a, depth=self.n_actions, dtype=tf.float32)
        self.q_fi_from_training_net_with_action = tf.reduce_sum(self.q_fi_from_training_net * a_one_hot, axis=-1)  #dot product


        # ------------------ build frozen_net ------------------
        self.batch_Nfi = tf.placeholder(tf.float32, shape=[None, self.w, self.h, self.m]) / 255  # input Next State
        if Train_Times == 0:
            self.frozen_name = 'col_0_frozen'
        elif Train_Times == 1:
            self.frozen_name = 'col_1_frozen'
        elif Train_Times == 2:
            self.frozen_name = 'col_2_frozen'
        col0_frozen_name = ['col_0_frozen', tf.GraphKeys.GLOBAL_VARIABLES]
        col0_frozen = build_collum(self.batch_Nfi, col0_frozen_name)
        col1_frozen_name = ['col_1_frozen', tf.GraphKeys.GLOBAL_VARIABLES]
        col1_frozen = build_collum(self.batch_Nfi, col1_frozen_name)
        col2_frozen_name = ['col_2_frozen', tf.GraphKeys.GLOBAL_VARIABLES]
        col2_frozen = build_collum(self.batch_Nfi, col2_frozen_name, h1s=[col0_frozen[0], col1_frozen[0]],
                                   h2s=[col0_frozen[1], col1_frozen[1]],
                                   h3s=[col0_frozen[2], col1_frozen[2]],
                                   h4s=[col0_frozen[3], col1_frozen[3]])
        if Train_Times == 0:
            self.q_Nfi_from_frozen_net = col0_frozen[4]
        elif Train_Times == 1:
            self.q_Nfi_from_frozen_net = col1_frozen[4]
        elif Train_Times == 2:
            self.q_Nfi_from_frozen_net = col2_frozen[4]

        self.q_fi_suppose_by_frozen_net = tf.placeholder(tf.float32, shape=[None, self.n_robot])
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.q_fi_suppose_by_frozen_net[:,0], self.q_fi_from_training_net_with_action[:,0]))
       
        train_vars = tf.get_collection(self.training_name)
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, var_list=train_vars)


    def store_transition(self, fi, a, r, Nfi, done):
        index = self.memory_counter % self.memory_size
        self.memory['fi'][index] = fi
        self.memory['a'][index] = a
        self.memory['r'][index] = r
        self.memory['Nfi'][index] = Nfi
        self.memory['done'][index] = done
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]  # [84,84,4] - > [1,84,84,4]
        if np.random.uniform() < self.exploration:  # exploration
            action = np.array([np.random.randint(0, self.n_actions), np.random.randint(0, self.n_actions),
                               np.random.randint(0, self.n_actions), np.random.randint(0, self.n_actions)])
        else:
            self.actions_value = self.sess.run(self.q_fi_from_training_net, feed_dict={self.batch_fi: observation})[0]
            action = np.argmax(self.actions_value, axis=-1)
        return action

    def learn(self):
        if self.train_step_counter % self.frozen_network_update_frequency == 0:
            t_params = tf.get_collection(self.frozen_name)
            e_params = tf.get_collection(self.training_name)
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
            self.saver.save(self.sess, SAVE_MODEL, global_step=self.train_step_counter)
            self.update_counter += 1
            
        if(self.exploration > self.final_exploration):
            self.exploration -= ( self.initial_exploration - self.final_exploration) / self.final_exploration_frame
        else:
            self.exploration = self.final_exploration

        
        if self.memory_counter > self.replay_start_size:
            # sample batch memory from all memory
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            #get q_fi_suppose_by_frozen_net
            q_Nfi_from_frozen_net = self.sess.run(self.q_Nfi_from_frozen_net, feed_dict={self.batch_Nfi: self.memory['Nfi'][sample_index]})
            done = self.memory['done'][sample_index]
            reward = self.memory['r'][sample_index]
            q_fi_suppose_by_frozen_net = np.zeros([self.batch_size, self.n_robot])
            for i in range(self.batch_size):
                if done[i] == True:
                    q_fi_suppose_by_frozen_net[i] = reward[i]
                else:
                    q_fi_suppose_by_frozen_net[i] = reward[i] + self.gamma * np.max(q_Nfi_from_frozen_net[i], axis=-1)

            # train training_network by q_fi_suppose_by_frozen_net
            _, self.outloss = self.sess.run([self._train_op, self.loss],
                feed_dict={self.q_fi_suppose_by_frozen_net : q_fi_suppose_by_frozen_net,
                            self.batch_fi: self.memory['fi'][sample_index],
                            self.batch_a: self.memory['a'][sample_index]})
            self.train_step_counter += 1
                            
        return self.actions_value, self.exploration,self.train_step_counter, self.update_counter,self.outloss
