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
from __future__ import absolute_import
import sys
import tensorflow as tf
import numpy as np
import random
import time
import math
# from robot_control_interface.env.gazebo_env.twoside_gazebo import twoside_gazebo
# from env.gazebo_env.suction_grasp_env import suction_grasp_env
from env.gazebo_env.suction_grasp_env import *
from ddpg import DDPG
from setting import SETTING
# import tf  as tf_trans #for euler <-> quaternion transformation


import copy
#from robot_control_interface.env.gazebo_env.handout_gazebo import handout_gazebo
#####################  hyper parameters  ####################

# MAX_EPISODES = 100000000
# MAX_EP_STEPS = 2500
# NEURONNUM = 500
# LR_A = 4e-5    # learning rate for actor
# LR_C = 8e-5    # learning rate for critic
# GAMMA = 0.9     # reward discount
# TAU = 0.001      # soft replacement
# MEMORY_CAPACITY = 20000
# BATCH_SIZE = 128

# RENDER = True
# ENV_NAME = 'Pendulum-v0'

###############################  training  ####################################


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [roll, pitch, yaw]


def main():
    # np.set_printoptions(precision=4, linewidth=125, suppress=True)
    # np.core.arrayprint._line_width = 125

    portStr = None
    if len(sys.argv) > 1:
        portStr = sys.argv[1]
    env = suction_grasp_env(portStr)
    env.interface.GAZEBO_SetModel("vs060",{},{},{})

    # while True:
    #     panel_pq = env.interface.GAZEBO_GetLinkPQByName("panel", "panel_link")
    #     tcp_pq = env.interface.GAZEBO_GetLinkPQByName("vs060", "J6")
    #     print("panel qp:",panel_pq)
    #     print("tcp_pq:", tcp_pq)
    #
    #     euler_panel = quaternion_to_euler(panel_pq['QX'],panel_pq['QY'],panel_pq['QZ'],panel_pq['QW'])
    #     # euler_tcp = quaternion_to_euler(tcp_pq['QX'],tcp_pq['QY'],tcp_pq['QZ'],tcp_pq['QW'])
    #
    #     euler_tcp = euler_panel
    #     euler_tcp[0] = np.random.rand(np.pi)
    #     euler_tcp[1] = np.random.rand(np.pi)
    #     euler_tcp[2] = 0.
    #
    #
    #     q_tcp = euler_to_quaternion(euler_tcp[0],euler_tcp[1],euler_tcp[2])
    #     tcp_pq["QX"] = q_tcp[0]
    #     tcp_pq["QY"] = q_tcp[1]
    #     tcp_pq["QZ"] = q_tcp[2]
    #     tcp_pq["QW"] = q_tcp[3]
    #
    #     env.interface.GAZEBO_SetLinkPQByName("vs060","J6",tcp_pq)
    #     env.interface.GAZEBO_Step(1)
    #     input("stop")

    # s_dim = env.retinfo()["obs_array_size"]
    # a_dim = env.retinfo()["act_array_size"]
    # a_bound = env.retinfo()["act_array_range"][1]
    state_dim = SETTING.STATE_SIZE
    act_dim = SETTING.ACT_SIZE
    act_bound = SETTING.ACT_BOUND

    ddpg = DDPG(act_dim, state_dim, act_bound)
    start_time = time.time()

    ddpg.initial_replay_memory(env)
    ddpg.train(env)

    # ddpg.restore_model()
    # ddpg.test(env, 100)
    # input("stop")


    # for i in range(SETTING.MAX_EPISODES):
    #     s, r, done, r_back = env.reset()
    #     # s, r, done, r_back = env.reset_from_record()
    #     reward_sum = 0
    #     if size_counter > 400:
    #         looptype = looptype + 1
    #         size_counter = 0
    #
    #     if looptype == 0:
    #         looptype = looptype + 1
    #
    #     if looptype == 4:
    #         if ddpg.index_pointer > SETTING.MEMORY_CAPACITY:
    #             looptype = 0
    #         else:
    #             looptype = 1
    #
    #     if looptype == 0 or looptype == 2:
    #         if looptype == 0:
    #             if random.random() < 0.5:
    #                 s_raw, r, done, r_back = env.reset()
    #                 # s_raw, r, done, r_back = env.reset_from_record()
    #
    #             else:
    #                 pass
    #         else:
    #             s_raw, r, done, r_back = env.reset()
    #             # s_raw, r, done, r_back = env.reset_from_record()
    #
    #     a_loss_sum = 0.0
    #     c_loss_sum = 0.0
    #     gerrorcount = 0
    #
    #
    #     # if i!=0 and i%1000 ==0:
    #     #     SETTING.LR_A /= 2.
    #     #     SETTING.LR_C /= 2.
    #     #     print("lr:{}".format(SETTING.LR_C))
    #
    #     q_list = []
    #     for j in range(SETTING.MAX_EP_STEPS):
    #         # Add exploration noise
    #         a = ddpg.choose_action(s)
    #         q = ddpg.get_Q_value(s)
    #         q_list.append(q)
    #
    #         if looptype != 0:
    #             a = np.clip(np.random.normal(a, explore_bound), -act_bound, act_bound)
    #         # s__back = env.get_back_s()["obs_array_data"][0]
    #         # s__raw, r, done = env.step(a)
    #         # s_ = s__raw["obs_array_data"][0]
    #
    #         s_, r, done = env.step(a)
    #         if looptype != 0:
    #             ddpg.store_transition(s, a, r / 1.0, done, s_,)
    #             size_counter = size_counter + 1
    #
    #             if ddpg.index_pointer > SETTING.MEMORY_CAPACITY:
    #                 # explore_bound *= .999    # decay the action randomness
    #                 a_loss, c_loss = ddpg.learn()
    #                 a_loss_sum = a_loss_sum + a_loss
    #                 c_loss_sum = c_loss_sum + c_loss
    #                 gerrorcount = gerrorcount + 1
    #         s = s_
    #         reward_sum += r
    #
    #         if j == SETTING.MAX_EP_STEPS - 1 or done == True:
    #             print('Episode:', i, 'Step:', j, ' Avg eward: %i' % int(reward_sum), 'Explore: %.5f' % explore_bound,
    #                   'A loss: %.6f' % (a_loss_sum / gerrorcount), 'TD error: %.6f' % (c_loss_sum / gerrorcount))
    #
    #             if looptype != 0:
    #                 if explore_bound < act_bound * 0.75:
    #                     explore_bound *= 1.0  # .9998
    #                 else:
    #                     explore_bound *= .9995
    #
    #                 if gerrorcount == 0:
    #                     gerrorcount = 1
    #                 print('Episode:', i, 'Step:', j, ' Avg eward: %i' % int(reward_sum), 'Explore: %.5f' % explore_bound,
    #                       'A loss: %.6f' % (a_loss_sum / gerrorcount), 'TD error: %.6f' % (c_loss_sum / gerrorcount))
    #
    #                 qLen = len(q_list)
    #                 for idx in range(qLen):
    #                     if idx > 15:
    #                         break
    #                     print(q_list[qLen-idx-1], end=" ")
    #                 print("")
    #
    #             break

    print('Running time: ', time.time() - start_time)



if __name__ == "__main__":
    main()










































