"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self):
        self.n_actions = 4 #up down left right

        self.batch_size = 32
        self.memory_size = 1000000  # replay memory size
        self.history_length = 4 #agent history length
        self.target_network_update_frequency = 10000 #target network update frequency
        self.gamma = 0.99  # discount factor
        self.action_repeat = 4
        self.update_frequency = 4
        self.initial_exploration = 1
        self.final_exploration = 0.1
        self.final_exploration_frame = 1000000
        self.reply_start_size = 50000
        #used by RMSProp
        self.learning_rate = 0.00025
        self.gredient_momentum = 0.95
        self.squared_gredient_momentum = 0.95
        self.min_squared_gradient = 0.01

        self._build_net()# consist of [target_net, evaluate_net]
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = [] #the error of every step

    def _build_net(self):

        # ------------------ build evaluate_net ------------------
        im = tf.placeholder(tf.float32, shape=[None, 84, 84, 4]) / 255  # [10,224,224,3]
        x = tf.placeholder(tf.float32, shape=[None, 1])
        y = tf.placeholder(tf.float32, shape=[None, 1])
        # i_s = tf.placeholder(tf.float32, shape=[1])

        keep_prob = tf.placeholder(tf.float32)

        col_eval_net = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        ## conv1 layer ##
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.1),collections = col_eval_net)
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]),collections = col_eval_net)
        conv1 = tf.nn.conv2d(im, W_conv1, strides=[1, 4, 4, 1], padding='SAME')
        h_conv1 = tf.nn.relu(conv1 + b_conv1)

        ## conv2 layer ##
        W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1),collections = col_eval_net)
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]),collections = col_eval_net)
        conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME')
        h_conv2 = tf.nn.relu(conv2 + b_conv2)

        ## conv3 layer ##
        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1),collections = col_eval_net)
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]),collections = col_eval_net)
        conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3 = tf.nn.relu(conv3 + b_conv3)

        # [n_samples, 6, 6, 64] ->> [n_samples, 2304]
        h_pool3_flat = tf.reshape(h_conv3, [-1, 2304])

        ## fc4 layer ##
        W_fc4 = tf.Variable(tf.truncated_normal([9216, 4096], stddev=0.1),collections = col_eval_net)
        b_fc4 = tf.Variable(tf.constant(0.1, shape=[4096]),collections = col_eval_net)
        h_fc4 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc4) + b_fc4)
        h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

        ## fc5 layer ##
        W_fc5 = tf.Variable(tf.truncated_normal([4096, 512], stddev=0.1),collections = col_eval_net)
        b_fc5 = tf.Variable(tf.constant(0.1, shape=[512]),collections = col_eval_net)
        h_fc5 = tf.matmul(h_fc4_drop, W_fc5) + b_fc5
        self.q_eval = h_fc5

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self._train_op = tf.train.RMSPropOptimizer(self.learning_rate,).minimize(self.loss)???

        # ------------------ build target_net ------------------
        im = tf.placeholder(tf.float32, shape=[None, 84, 84, 4]) / 255  # [10,224,224,3]
        x = tf.placeholder(tf.float32, shape=[None, 1])
        y = tf.placeholder(tf.float32, shape=[None, 1])
        # i_s = tf.placeholder(tf.float32, shape=[1])

        keep_prob = tf.placeholder(tf.float32)

        col_targ_net = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        ## conv1 layer ##
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.1),collections = col_targ_net)
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]),collections = col_targ_net)
        conv1 = tf.nn.conv2d(im, W_conv1, strides=[1, 4, 4, 1], padding='SAME')
        h_conv1 = tf.nn.relu(conv1 + b_conv1)

        ## conv2 layer ##
        W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1),collections = col_targ_net)
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]),collections = col_targ_net)
        conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME')
        h_conv2 = tf.nn.relu(conv2 + b_conv2)

        ## conv3 layer ##
        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1),collections = col_targ_net)
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]),collections = col_targ_net)
        conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3 = tf.nn.relu(conv3 + b_conv3)

        # [n_samples, 6, 6, 64] ->> [n_samples, 2304]
        h_pool3_flat = tf.reshape(h_conv3, [-1, 2304])

        ## fc4 layer ##
        W_fc4 = tf.Variable(tf.truncated_normal([9216, 4096], stddev=0.1),collections = col_targ_net)
        b_fc4 = tf.Variable(tf.constant(0.1, shape=[4096]),collections = col_targ_net)
        h_fc4 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc4) + b_fc4)
        h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

        ## fc5 layer ##
        W_fc5 = tf.Variable(tf.truncated_normal([4096, 512], stddev=0.1),collections = col_targ_net)
        b_fc5 = tf.Variable(tf.constant(0.1, shape=[512]),collections = col_targ_net)
        h_fc5 = tf.matmul(h_fc4_drop, W_fc5) + b_fc5
        self.q_next = h_fc5

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) #horizontally stack

        index = self.memory_counter % self.memory_size # replace the old memory with new memory
        self.memory[index, :] = transition #override the old memory

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon: #mao si fan le
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
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('target_params_rplaced')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

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
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()