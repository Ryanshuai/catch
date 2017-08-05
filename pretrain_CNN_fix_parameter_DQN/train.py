import hunter_agent as HA
import tensorflow as tf
from collections import Counter
import numpy as np
import cv2
from game import wrapped_flappy_bird as bird

counter = Counter({'total_steps':0,'train_steps':0,'episode':0,'step_in_episode':0,
                   'r_sum_in_episode':0,'loss':0,'exploration':0,'summary':0})
num_episodes = 50*1000
update_freq = 4
num_before_train=1000
save_mode_per_episode = 1000
action_num = 2

tf.reset_default_graph()

env = bird.GameState()

training_net = HA.TrainingQNetwork(act_num=action_num)
frozen_net = HA.FrozenQNetwork(act_num=action_num)
memory = HA.ExperienceMemory()
trainer = HA.Trainer(training_net,frozen_net,memory)
chooser = HA.Chooser(act_num=action_num,num_before_train=num_before_train)
model = HA.Model()
updater = HA.Updater()


def next_step(a):
    action = np.zeros(shape=[2, ])
    action[a] = 1
    nextObservation = np.zeros(shape=[84, 84, 4], dtype = np.uint8)
    reward = 0
    reward_sum = 0
    terminal = False
    for i in range(4):
        next_image, reward, terminal = env.frame_step(action)
        reward_sum += reward
        # terminal = True, flappyBird is inited automatically
        if terminal:
            break
        next_image = cv2.cvtColor(cv2.resize(next_image, (84, 84)), cv2.COLOR_BGR2GRAY)
        nextObservation[:, :, i] = next_image
    return nextObservation, reward_sum , terminal


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    tf_writer = tf.summary.FileWriter('logs/')
    tf_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    model.restore(sess)
    updater.init_frozen_net(sess)
    for episode in range(num_episodes):
        counter.update(('episode',))
        fi, r, done = next_step(0)

        counter['step_in_episode'] = 0
        counter['r_sum_in_episode'] = 0
        while not done:
            a,counter['exploration'] = chooser.choose_action(sess,training_net,fi,counter['total_steps'])
            Nfi,r,done = next_step(a)
            counter.update(('total_steps',))
            memory.add(fi,a,r,Nfi,done)
            fi = Nfi
            counter.update(('step_in_episode',))
            counter['r_sum_in_episode'] += r

            if counter['total_steps'] > num_before_train and counter['total_steps'] % update_freq == 0:
                counter.update(('summary',))
                if counter['summary']%10 == 0:
                    counter['loss'],merged_summary = trainer.train_traing_net_with_summary(sess)
                    tf_writer.add_summary(merged_summary,counter['train_steps'])
                else:
                    counter['loss'] = trainer.train_traing_net(sess)
                counter.update(('train_steps',))
                updater.update_frozen_net(sess)
                print('---------------------------------------------------------------------------------train_steps:', counter['train_steps'],'loss:', '%.8f' % counter['loss'])

        print('---------------------------------------------------------------------------------total_steps:',counter['total_steps'],'episode:',counter['episode'],'exploration:','%.3f'%counter['exploration'],'r_sum_in_episode:',counter['r_sum_in_episode'])

        if counter['episode'] % save_mode_per_episode == 0:
            model.store(sess, counter['episode'])

