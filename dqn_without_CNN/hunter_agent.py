import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

#Implementing the network itself

class FrozenQNetwork():
    def __init__(self,act_num):
        self.collection_pretrain = ['pretrain_frozen_variable', tf.GraphKeys.GLOBAL_VARIABLES]
        self.collection = ['frozen_variable', tf.GraphKeys.GLOBAL_VARIABLES]

        xavier_init = tf.contrib.layers.xavier_initializer()

        self.robots_state = tf.placeholder(shape=[None, 16], dtype=tf.float32)  # [bs,40]

        ## fc1 layer ##
        self.W_fc1 = tf.Variable(xavier_init([16, 1280]), collections=self.collection_pretrain)
        self.b_fc1 = tf.Variable(tf.constant(0., shape=[1280]), collections=self.collection_pretrain)
        self.h_fc1 = tf.nn.relu(tf.matmul(self.robots_state, self.W_fc1) + self.b_fc1)#[bs, 1024]
        ## fc2 layer ##
        self.W_fc2 = tf.Variable(xavier_init([1280, 1280]),collections=self.collection_pretrain)
        self.b_fc2 = tf.Variable(tf.constant(0., shape=[1280]), collections=self.collection_pretrain)
        self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)#[bs, 64]

        ## fc3 layer ##
        self.W_fc3 = tf.Variable(xavier_init([1280, 64]), collections=self.collection_pretrain)
        self.b_fc3 = tf.Variable(tf.constant(0., shape=[64]), collections=self.collection_pretrain)
        self.h_fc3 = tf.nn.relu(tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3) # [bs, 4]

        self.streamA, self.streamV = tf.split(self.h_fc3, 2, 1)#[bs, 32],#[bs, 32]

        self.AW = tf.Variable(xavier_init([32,act_num]),collections=self.collection) #[32,act_num]
        self.VW = tf.Variable(xavier_init([32,1]),collections=self.collection) #[32,1]
        self.Advantage = tf.matmul(self.streamA,self.AW) #[bs,act_num]
        self.Value = tf.matmul(self.streamV,self.VW) #[bs,1]
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))#[bs,act_num]


class TrainingQNetwork():
    def __init__(self, act_num):
        self.collection = ['training_variable', tf.GraphKeys.GLOBAL_VARIABLES]
        self.collection_pretrain = ['pretrain_training_variable', tf.GraphKeys.GLOBAL_VARIABLES]

        xavier_init = tf.contrib.layers.xavier_initializer()

        self.robots_state = tf.placeholder(shape=[None, 16], dtype=tf.float32)  # [bs,40]

        ## fc1 layer ##
        self.W_fc1 = tf.Variable(xavier_init([16, 1280]), collections=self.collection_pretrain)
        self.b_fc1 = tf.Variable(tf.constant(0., shape=[1280]), collections=self.collection_pretrain)
        self.h_fc1 = tf.nn.relu(tf.matmul(self.robots_state, self.W_fc1) + self.b_fc1)#[bs, 1024]
        tf.summary.histogram('W_fc1', self.W_fc1)
        tf.summary.histogram('b_fc1', self.b_fc1)
        ## fc2 layer ##
        self.W_fc2 = tf.Variable(xavier_init([1280, 1280]),collections=self.collection_pretrain)
        self.b_fc2 = tf.Variable(tf.constant(0., shape=[1280]), collections=self.collection_pretrain)
        self.h_fc2 = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2#[bs, 64]
        tf.summary.histogram('W_fc2', self.W_fc2)
        tf.summary.histogram('b_fc2', self.b_fc2)
        ## fc3 layer ##
        self.W_fc3 = tf.Variable(xavier_init([1280, 64]), collections=self.collection_pretrain)
        self.b_fc3 = tf.Variable(tf.constant(0., shape=[64]), collections=self.collection_pretrain)
        self.h_fc3 = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3 # [bs, 4]
        tf.summary.histogram('W_fc3', self.W_fc3)
        tf.summary.histogram('b_fc3', self.b_fc3)

        self.streamA, self.streamV = tf.split(self.h_fc3, 2, 1)  # [bs, 32],#[bs, 32]

        self.AW = tf.Variable(xavier_init([32, act_num]), collections=self.collection)  # [32,act_num]
        self.VW = tf.Variable(xavier_init([32, 1]), collections=self.collection)  # [32,1]
        tf.summary.histogram('AW', self.AW)
        tf.summary.histogram('VW', self.VW)
        self.Advantage = tf.matmul(self.streamA, self.AW)  # [bs,act_num]
        self.Value = tf.matmul(self.streamV, self.VW)  # [bs,1]
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))  # [bs,act_num]
        self.predict = tf.argmax(self.Qout, 1)  # [bs]

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)  # bs
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)  # bs
        self.actions_onehot = tf.one_hot(self.actions, act_num, dtype=tf.float32)  #[bs,act_num]

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)  # bs

        self.td_error = tf.square(self.targetQ - self.Q)  # bs
        self.loss = tf.reduce_mean(self.td_error)  # 1
        tf.summary.scalar('loss',self.loss)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0005)
        self.optimize = self.trainer.minimize(self.loss)
        self.merged_summary = tf.summary.merge_all()


class ExperienceMemory():
    def __init__(self, memory_size=20000):
        self.memory = []
        self.memory_size = memory_size

    def add(self, fi, a, r, Nfi, done):

        experience = np.reshape(np.array([fi, a, r, Nfi, done]), [1, 5])
        if len(self.memory) + len(experience) >= self.memory_size:
            self.memory[0:(len(experience) + len(self.memory)) - self.memory_size] = []
        self.memory.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.memory, size)), [size, 5])


class Model():
    def __init__(self,path="./model"):
        self.saver = tf.train.Saver()
        self.path = path

    def store(self,sess,episode):

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.saver.save(sess, self.path + '/model.cptk', global_step=episode)

    def restore(self,sess):
        if not os.path.exists(self.path):
            print('no Model')
        else:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(self.path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)


class Chooser():
    def __init__(self,act_num,num_before_train):
        self.act_num = act_num
        self.initial_exploration = 1.  # 1. #initial
        self.final_exploration = 0.1
        self.e = self.initial_exploration
        self.final_exploration_frame = 30000
        self.num_before_train=num_before_train

    def choose_action(self,sess,training_net,fi,total_step):
        if total_step>self.num_before_train:
            if (self.e > self.final_exploration):
                self.e -= (self.initial_exploration - self.final_exploration) / self.final_exploration_frame
            else:
                self.e = self.final_exploration

        if np.random.rand(1) < self.e or total_step < self.num_before_train:
            a = np.random.randint(0, self.act_num)
            print('rand_act')
        else:
            act,act_value = sess.run([training_net.predict,training_net.Qout], feed_dict={training_net.robots_state: [fi]})
            print('act_value:',['%.6f' %i for i in act_value[0]])
            a = act[0]
        return a,self.e


class Trainer():
    def __init__(self,training_net,frozen_net,memory,batch_size = 64):
        self.bs = batch_size
        self.gamma = 0.99
        self.trn_net = training_net
        self.fro_net = frozen_net
        self.mem = memory

    def train_traing_net(self,sess):
        trainBatch = self.mem.sample(self.bs)
        # Below we perform the Double-DQN update to the target Q-values
        Q_by_frozen_net = sess.run(self.fro_net.Qout, feed_dict={self.fro_net.robots_state: np.vstack(trainBatch[:, 3])})#[bs,act_num]
        argmax = sess.run(self.trn_net.predict,feed_dict={self.trn_net.robots_state: np.vstack(trainBatch[:, 3])}) # [bs]
        end_multiplier = -(trainBatch[:, 4] - 1)#array([bs])
        doubleQ = Q_by_frozen_net[range(self.bs), argmax]#array([bs])
        targetQ = trainBatch[:, 2] + (self.gamma * doubleQ * end_multiplier) #array([bs])
        #Update the network with our target values.
        _, loss = sess.run(
            [self.trn_net.optimize, self.trn_net.loss],
            feed_dict={self.trn_net.robots_state: np.vstack(trainBatch[:, 0]),
                       self.trn_net.targetQ: targetQ,
                       self.trn_net.actions: trainBatch[:, 1]})
        return loss

    def train_traing_net_with_summary(self,sess):
        trainBatch = self.mem.sample(self.bs)
        # Below we perform the Double-DQN update to the target Q-values
        Q_by_frozen_net = sess.run(self.fro_net.Qout, feed_dict={self.fro_net.robots_state: np.vstack(trainBatch[:, 3])})#[bs,act_num]
        argmax = sess.run(self.trn_net.predict,feed_dict={self.trn_net.robots_state: np.vstack(trainBatch[:, 3])}) # [bs]
        end_multiplier = -(trainBatch[:,4] - 1)#array([bs])
        doubleQ = Q_by_frozen_net[range(self.bs), argmax]#array([bs])
        targetQ = trainBatch[:, 2] + (self.gamma * doubleQ * end_multiplier) #array([bs])
        #Update the network with our target values.
        _, loss, merged_summary = sess.run(
            [self.trn_net.optimize, self.trn_net.loss, self.trn_net.merged_summary],
            feed_dict={self.trn_net.robots_state: np.vstack(trainBatch[:, 0]),
                       self.trn_net.targetQ: targetQ,
                       self.trn_net.actions: trainBatch[:, 1]})
        return loss,merged_summary


class Updater():
    def __init__(self):
        ur = 0.001
        frozen_params = tf.get_collection('frozen_variable')
        training_params = tf.get_collection('training_variable')
        self.init_op_holder = []
        for idx, var in enumerate(frozen_params):
            op = frozen_params[idx].assign(training_params[idx].value())
            self.init_op_holder.append(op)

        self.op_holder = []
        for idx, var in enumerate(frozen_params):
            op = frozen_params[idx].assign((1 - ur) * var.value() + ur * training_params[idx].value())
            self.op_holder.append(op)

    def init_frozen_net(self,sess):
        for op in self.init_op_holder:
            sess.run(op)

    def update_frozen_net(self,sess):
        for op in self.op_holder:
            sess.run(op)