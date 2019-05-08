
import numpy as np
import tensorflow as tf
from setting import SETTING
import time
from env.gazebo_env.suction_grasp_env import *
from rrt import RRT

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        tf.reset_default_graph()    #In order to use multistage model

        self.memory = np.zeros((SETTING.MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 + 1), dtype=np.float32)
        self.index_pointer = 0
        # if SETTING.RESET_FROM_LOG:
        #     self.log_file = open("0424_log_secondStep.txt","w+")
        # else:
        #     self.log_file = open("0424_log_allStep.txt","w+")

        # session_config = tf.ConfigProto()
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        session_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=session_config)
        self.saved_model_path = "/home/ccchang/ddpg_refactor/saved_model/"
        self.max_goal_rate = 1

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        with tf.device('/GPU:0'):
            self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
            self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
            self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')
            self.done = tf.placeholder(tf.float32, [None, 1],'done')

            self.a = self._build_a(self.S,)
            self.q = self._build_c(self.S, self.a)
            a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
            c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
            ema = tf.train.ExponentialMovingAverage(decay=1 - SETTING.TAU)          # soft replacement
            target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
            a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
            q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

            self.a_loss = - tf.reduce_mean(self.q)  # maximize the q
            self.actor_train = tf.train.AdamOptimizer(SETTING.LR_A).minimize(self.a_loss, var_list=a_params)

            with tf.control_dependencies(target_update):    # soft targetnet update
                # q_target = self.R + SETTING.GAMMA * q_
                q_target = self.R + (1.-self.done) * SETTING.GAMMA * q_

                self.td_loss = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
                self.critic_train = tf.train.AdamOptimizer(SETTING.LR_C).minimize(self.td_loss, var_list=c_params)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

    def save_model(self, sName):
        save_path = self.saver.save(self.sess, self.saved_model_path + sName)
        print("Model saved in path: %s" % save_path)

    def restore_model(self, path=None):
        if path != None:
            self.saver.restore(self.sess, path)
        else:
            self.saver.restore(self.sess, self.saved_model_path)
        print("Model restored.")
        return

    def initial_replay_memory(self, env):
        size_counter = 0
        act_bound = SETTING.ACT_BOUND
        explore_bound = act_bound * 1.5
        for i in range(1000):
            if SETTING.RESET_FROM_LOG:
                s, r, done = env.reset_from_record()
            elif SETTING.RESET_FROM_RRT:
                s, r, done = env.reset_from_RRT()
            else:
                s, r, done = env.reset()

            print("current memory size:", size_counter)

            if size_counter >= SETTING.MEMORY_CAPACITY:
                break
            for j in range(SETTING.MAX_EP_STEPS):
                a = self.choose_action(s)
                a = np.clip(np.random.normal(a, explore_bound), -act_bound, act_bound)
                s_, r, done, is_goal = env.step(a)
                self.store_transition(s, a, r / 1.0, done, s_)
                s = s_
                size_counter = size_counter + 1
                if j == SETTING.MAX_EP_STEPS - 1 or done == True:
                    break
        return

    def train(self,env):
        self.log_file = open("0507_ddpgRRT_secondStep", "w+")
        for i in range(SETTING.MAX_EPISODES):
            if i%100 == 0 and i!=0:
                self.test(env,20)
                # self.save_model()
            if SETTING.RESET_FROM_LOG:
                s, r, done = env.reset_from_record()
            elif SETTING.RESET_FROM_RRT:
                s, r, done = env.reset_from_RRT()
            else:
                s, r, done = env.reset()
            a_loss_sum = 0
            c_loss_sum = 0
            reward_sum = 0
            act_bound = SETTING.ACT_BOUND
            explore_bound = act_bound*1.5
            for j in range(SETTING.MAX_EP_STEPS):
                a = self.choose_action(s)
                a = np.clip(np.random.normal(a, explore_bound), -act_bound, act_bound)
                s_, r, done, is_goal = env.step(a)
                self.store_transition(s, a, r / 1.0, done, s_,)
                a_loss, c_loss = self.learn()
                a_loss_sum = a_loss_sum + a_loss
                c_loss_sum = c_loss_sum + c_loss
                s = s_
                reward_sum += r
                if j == SETTING.MAX_EP_STEPS - 1 or done == True:
                    print('Episode:', i, 'Step:', j, ' Avg reward: %i' % int(reward_sum), 'Explore: %.5f' % explore_bound,
                          'A loss: %.6f' % (a_loss_sum / (j+1)), 'TD error: %.6f' % (c_loss_sum / (j+1)))
                    break

    def test(self, env, test_round):
        goal_count = 0
        step_count = 0
        reward_sum = 0
        act_bound = SETTING.ACT_BOUND

        end_pos = [1.57, 0.96, 0.828, 0., 1.36, 0.] # first step end
        end_pos2 = [1.59, -1., -0.575, 0., -1.56, 0.] # second step end , horizontally move
        end_pos3 = [4.705, 1.024, 0.59, 0, 1.54, 0.] #second step end, vertically move

        max_iteration = 600
        g_rate = 0.05
        e_factor = 0.1

        for i in range(test_round):
            if SETTING.RESET_FROM_LOG:
                s, r, done = env.reset_from_record()
            elif SETTING.RESET_FROM_RRT:
                s, r, done = env.reset_from_RRT()
            else:
                s, r, done = env.reset()

            for j in range(SETTING.MAX_EP_STEPS):
                a = self.choose_action(s)
                a = np.clip(a, -act_bound, act_bound)
                s_, r, done, is_goal = env.step(a)
                time.sleep(0.04)
                s = s_
                if is_goal:
                    goal_count += 1
                if j == SETTING.MAX_EP_STEPS - 1 or done == True:
                    reward_sum += r
                    step_count += j
                    break

                #Apply RRT
                if SETTING.RESET_FROM_RRT:
                    dist = env.get_distance(link1_name="J6", link2_name="vg2_link")
                else:
                    dist = env.get_distance(link1_name="J6", link2_name="panel_link")

                print("dist:",dist)

                if dist < 0.15:
                    start_pos = joint_dict_to_list(env.interface.GAZEBO_GetPosition("vs060"))
                    # print("start pos:", start_pos)
                    # input("stop")
                    if SETTING.RESET_FROM_RRT:
                        rrt = RRT(env=env, start=start_pos, goal=end_pos2, goal_sample_rate=g_rate, expand_factor=e_factor,
                              max_iteration=max_iteration)
                    else:
                        rrt = RRT(env=env, start=start_pos, goal=end_pos, goal_sample_rate=g_rate,expand_factor=e_factor,
                                  max_iteration=max_iteration)
                    ptr = None
                    while ptr == None:
                        ptr = rrt.plan(animation=False)

                    while ptr != None:
                        env.interface.GAZEBO_SetModel("vs060", make_6joint_dict(ptr.joint_pos), {}, {})
                        time.sleep(0.04)
                        env.interface.GAZEBO_Step(1)
                        ptr = ptr.next
                    goal_count += 1
                    break

        goal_rate = round(goal_count/test_round*100, 2)
        if SETTING.SAVE_MODEL:
            if self.max_goal_rate <= goal_rate:
                self.max_goal_rate = goal_rate
                self.save_model("0508_ddpgRRT_secondStep" + str(goal_rate) + ".ckpt")

                # if SETTING.RESET_FROM_LOG==True:
                #     self.save_model("0424_secondStep" + str(goal_rate) + ".ckpt")
                # else:
                #     self.save_model("0424_firstStep" + str(goal_rate) + ".ckpt")
            print("{}\t{}\t{}".format(goal_rate, reward_sum / test_round, step_count), file=self.log_file)
            self.log_file.flush()
        print("===============================================================================")
        print("goal_rate:{}% average_reward:{}, step_count:{}"
                  .format(goal_rate, reward_sum/test_round, step_count))

        return


    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def get_Q_value(self, s):
        return self.sess.run(self.q, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(SETTING.MEMORY_CAPACITY, size=SETTING.BATCH_SIZE)
        bt = self.memory[indices, :]
        # bs = bt[:, :self.s_dim]
        # ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        # br = bt[:, -self.s_dim - 1: -self.s_dim]
        # bs_ = bt[:, -self.s_dim:]

        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim : self.s_dim + self.a_dim +1]
        bdone = bt[:, self.s_dim + self.a_dim +1 : self.s_dim + self.a_dim + 2]
        bs_ = bt[:, -self.s_dim:]

        # print("shape of bt:",np.shape(bt))
        # print("shape of bs:",np.shape(bs))
        # print("shape of ba:",np.shape(ba))
        # print("shape of br:",np.shape(br))
        # print("shape of bdone:",np.shape(bdone))
        # print("shape of bs_:",np.shape(bs_))
        # input("stop here")

        _, a_loss = self.sess.run([self.actor_train, self.a_loss], {self.S: bs})
        _, td_error = self.sess.run([self.critic_train, self.td_loss],
                                    {self.S: bs, self.a: ba, self.R: br, self.done: bdone, self.S_: bs_})

        return a_loss, td_error

    def store_transition(self, s, a, r, done, s_):
        transition = np.hstack((s, a, [r], done, s_))
        self.memory[self.index_pointer, :] = transition
        self.index_pointer += 1
        if self.index_pointer >= SETTING.MEMORY_CAPACITY:
            self.index_pointer %= SETTING.MEMORY_CAPACITY

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None   else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, SETTING.NEURONNUM, activation=tf.nn.relu, name='l1', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, SETTING.NEURONNUM, activation=tf.nn.relu, name='l2', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, SETTING.NEURONNUM, activation=tf.nn.relu, name='l3', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, SETTING.NEURONNUM, activation=tf.nn.relu, name='l4', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            # net = tf.layers.dense(net, SETTING.NEURONNUM, activation=tf.nn.relu, name='l5', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())

            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = SETTING.NEURONNUM
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            net = tf.layers.dense(net, SETTING.NEURONNUM, activation=tf.nn.relu, name='c1', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, SETTING.NEURONNUM, activation=tf.nn.relu, name='c2', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, SETTING.NEURONNUM, activation=tf.nn.relu, name='c3', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            # net = tf.layers.dense(net, SETTING.NEURONNUM, activation=tf.nn.relu, name='c4', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())


            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)