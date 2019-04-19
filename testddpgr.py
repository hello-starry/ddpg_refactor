"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np

import time
from robot_control_interface.env.gazebo_env.twoside_gazebo import twoside_gazebo
#from robot_control_interface.env.gazebo_env.handout_gazebo import handout_gazebo
#####################  hyper parameters  ####################

MAX_EPISODES = 100000
MAX_EP_STEPS = 800

NEURONNUM = 500
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128

RENDER = True
ENV_NAME = 'Pendulum-v0'


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=session_config)
        #self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        self.a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        _, a_loss = self.sess.run([self.atrain, self.a_loss], {self.S: bs})
        _, td_error = self.sess.run([self.ctrain, self.td_error], {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        return a_loss, td_error

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, NEURONNUM, activation=tf.nn.relu, name='l1', trainable=trainable)

            net = tf.layers.dense(net, NEURONNUM, activation=tf.nn.relu, name='l2', trainable=trainable)
            net = tf.layers.dense(net, NEURONNUM, activation=tf.nn.relu, name='l3', trainable=trainable)
            net = tf.layers.dense(net, NEURONNUM, activation=tf.nn.relu, name='l4', trainable=trainable)
            net = tf.layers.dense(net, NEURONNUM, activation=tf.nn.relu, name='l5', trainable=trainable)


            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = NEURONNUM
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            net = tf.layers.dense(net, NEURONNUM, activation=tf.nn.relu, name='c1', trainable=trainable)
            net = tf.layers.dense(net, NEURONNUM, activation=tf.nn.relu, name='c2', trainable=trainable)
            net = tf.layers.dense(net, NEURONNUM, activation=tf.nn.relu, name='c3', trainable=trainable)
            net = tf.layers.dense(net, NEURONNUM, activation=tf.nn.relu, name='c4', trainable=trainable)


            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################

teste = twoside_gazebo()
#teste = handout_gazebo()
s_dim = teste.retinfo()["obs_array_size"]
a_dim = teste.retinfo()["act_array_size"]
a_bound = teste.retinfo()["act_array_range"][1]
ddpg = DDPG(a_dim, s_dim, a_bound)

var = a_bound * 1.5  # control exploration
t1 = time.time()

looptype = 0





for i in range(MAX_EPISODES):
    s_raw, r, done = teste.reset()
    s = s_raw["obs_array_data"][0]
    ep_reward = 0

    looptype = looptype + 1
    if looptype == 4:
        if ddpg.pointer > MEMORY_CAPACITY:
            looptype = 0
        else:
            looptype = 1

    galoss = 0.0
    gtderror = 0.0
    gerrorcount = 0

    for j in range(MAX_EP_STEPS):




        # Add exploration noise
        a = ddpg.choose_action(s)
        if looptype != 0:
            a = np.clip(np.random.normal(a, var), -a_bound, a_bound)    # add randomness to action selection for exploration

        s__raw, r, done = teste.step(a)
        s_ = s__raw["obs_array_data"][0]


        #print(looptype)

        if looptype != 0:
            ddpg.store_transition(s, a, r / 1.0, s_)
            if ddpg.pointer > MEMORY_CAPACITY:
                #var *= .999    # decay the action randomness
                a_loss, td_error = ddpg.learn()
                galoss = galoss + a_loss
                gtderror = gtderror + td_error
                gerrorcount = gerrorcount + 1



        s = s_
        ep_reward += r


        if j == MAX_EP_STEPS-1 or done == True:
            if looptype != 0:
                if var < a_bound * 0.75:
                    var *= .9998
                else:
                    var *= .9995

                if gerrorcount == 0:
                    gerrorcount = 1
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.5f' % var, 'A loss: %.6f' % (galoss / gerrorcount), 'TD error: %.6f' % (gtderror / gerrorcount))
            # if ep_reward > -300:RENDER = True
            break

print('Running time: ', time.time() - t1)