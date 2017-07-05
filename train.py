import hunter_agent as HA
import tensorflow as tf
from collections import Counter

counter = Counter({'total_steps':0,'train_steps':0,'episode':0,'step_in_episode':0,'r_sum_in_episode':0})
num_episodes = 10*1000
max_step_in_one_episode = 100
update_freq = 4
num_pre_train=1000
save_mode_every = 1000

tf.reset_default_graph()

training_net = HA.TrainingQNetwork(act_num=5)
frozen_net = HA.FrozenQNetwork(act_num=5)
memory = HA.ExperienceMemory()
model = HA.Model()
chooser = HA.Chooser(num_pre_train=num_pre_train)
updater = HA.Updater()
ploter = HA.Ploter()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.restore(sess)
    for episode in range(num_episodes):
        counter.update(('episode',))
        fi = env.reset()
        done = False

        counter['step_in_episode'] = 0
        counter['r_sum_in_episode'] = 0
        while counter['step_in_episode'] < max_step_in_one_episode:
            a = chooser.choose_action(sess,training_net,fi,counter['total_steps'])
            Nfi,r,done = env.step(a)
            counter.update(('total_steps',))
            memory.add(fi,a,r,Nfi,done)
            fi = Nfi
            counter.update(('step_in_episode',))
            counter['r_sum_in_episode'] += r

            if counter['total_steps'] > num_pre_train and counter['total_steps'] % update_freq == 0:
                HA.train_traing_net(sess,training_net,frozen_net,memory)
                counter.update(('train_steps',))
                updater.update_frozen_net(sess)

            if done==True:
                break

        ploter.save(counter['r_sum_in_episode'])

        if counter['episode'] % save_mode_every == 0:
            model.store(sess,counter['episode'])
