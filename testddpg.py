import tensorflow as tf
import numpy as np
import time
import random
from robot_control_interface.env.gazebo_env.pullout_gazebo import pullout_gazebo
#####################  hyper parameters  ####################

MAX_EPISODES = 50000
MAX_EP_STEPS = 125
LR_A = 0.0004    # learning rate for actor
LR_C = 0.0004    # learning rate for critic
GAMMA = 0.995     # reward discount
TAU = 0.0001      # soft replacement
MEMORY_CAPACITY = 5000
BATCH_SIZE = 64
MIDDLE = 100
RENDER = True
ENV_NAME = 'Hopper-v2'
#FetchPickAndPlace-v0
#HalfCheetah-v2 FetchPickAndPlace-v0 InvertedDoublePendulum-v2 Hopper-v2



###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

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

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

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

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, MIDDLE, activation=tf.nn.relu, name='l1', trainable=trainable)

            net = tf.layers.dense(net, MIDDLE, activation=tf.nn.relu, name='l2', trainable=trainable)
            net = tf.layers.dense(net, MIDDLE, activation=tf.nn.relu, name='l3', trainable=trainable)
            net = tf.layers.dense(net, MIDDLE, activation=tf.nn.relu, name='l4', trainable=trainable)

            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = MIDDLE
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            net = tf.layers.dense(net, MIDDLE, activation=tf.nn.relu, name='c2', trainable=trainable)
            net = tf.layers.dense(net, MIDDLE, activation=tf.nn.relu, name='c3', trainable=trainable)
            net = tf.layers.dense(net, MIDDLE, activation=tf.nn.relu, name='c4', trainable=trainable)

            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################
teste = pullout_gazebo()
s_dim = teste.retinfo()["obs_array_size"]
a_dim = teste.retinfo()["act_array_size"]
a_bound = teste.retinfo()["act_array_range"][1]


ddpg = DDPG(a_dim, s_dim, a_bound)

var = a_bound * 3.  # control exploration
t1 = time.time()
trandom = False


tmplist = list()
thresh = 20.0
for i in range(MAX_EPISODES):
    s_raw, r, done = teste.reset()
    s = s_raw["obs_array_data"][0]


    ep_reward = 0.0
    if random.random() < 0.3:
        trandom = True
    else:
        trandom = False    
    trandomstartstep = random.randint(int(-MAX_EP_STEPS / 25), int(MAX_EP_STEPS / 7))
    steps = 0        
    for j in range(MAX_EP_STEPS):
        steps = steps + 1
        
        if ddpg.pointer < MEMORY_CAPACITY:
            a = np.random.random((a_dim))
            a = a - 0.5
            a = a * a_bound * 2.0              
        else:
            if trandom:
                if steps < trandomstartstep:
                    #print("1", trandom, steps, trandomstartstep)
                    a = ddpg.choose_action(s)
                    #print(a, end="---")
                    a = np.clip(np.random.normal(a, var), -a_bound, a_bound)    # add randomness to action selection for exploration
                    #print(a, end="---")                    
                else:
                    #print("2", trandom, steps, trandomstartstep)
                    a = np.random.random((a_dim))
                    a = a - 0.5
                    a = a * a_bound * 2.0
            else:
                #print("3", trandom, steps, trandomstartstep)
                a = ddpg.choose_action(s)

                #print(a, end="---")
                a = np.clip(np.random.normal(a, var), -a_bound, a_bound)    # add randomness to action selection for exploration
                #print(a, end="---")


        
        #print(a)
        s__raw, r, done = teste.step(a)
        s_ = s__raw["obs_array_data"][0]
        

        #"""
        #print(s_[0], r)
        #r = r + (s_[0] - 0.7) / 2.0
        #"""
        tmplist.append([s, a, r / 1.0, s_])
        
        #ddpg.store_transition(s, a, r / 1.0, s_)

        #if ddpg.pointer > MEMORY_CAPACITY:
            #var *= .999998    # decay the action randomness
            #var *= .99997
            #ddpg.learn()

        s = s_.copy()
        ep_reward += r

        #print("**********")
        if j == MAX_EP_STEPS-1 or done == True:# or (done == True and steps >= MAX_EP_STEPS / 2):
            
            ifadd = False
            if ddpg.pointer > MEMORY_CAPACITY:
                if steps > thresh:
                    ifadd = True
            else:
                if steps > thresh:
                    ifadd = True
            if ifadd:
                thresh = thresh + 0.025
                print("added!", end="---")      
                for ti in range(len(tmplist)):
                    ddpg.store_transition(tmplist[ti][0], tmplist[ti][1], tmplist[ti][2], tmplist[ti][3])
                if ddpg.pointer > MEMORY_CAPACITY:
                    print("train!", end="---")      
                    var *= .999
                    for _ in range(20):
                        ddpg.learn()
                
            tmplist.clear()    


            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, "steps:", steps)
            # if ep_reward > -300:RENDER = True
            break

print('Running time: ', time.time() - t1)