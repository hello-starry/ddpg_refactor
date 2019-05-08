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

# CUDA_VISIBLE_DEVICES=1

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


def multistage_test(env, test_round, ddpg1, ddpg2):
    goal_count = 0
    step_count = 0
    reward_sum = 0
    act_bound = SETTING.ACT_BOUND
    for i in range(test_round):
        s, r, done = env.reset()
        for j in range(SETTING.MAX_EP_STEPS):
            if env.is_sucked == False:
                a = ddpg1.choose_action(s)
            else:
                a = ddpg2.choose_action(s)

                # print("output1:", ddpg1.choose_action(s))
                # print("output2:", ddpg2.choose_action(s))
                # input("stop")

            a = np.clip(a, -act_bound, act_bound)
            s_, r, done, is_goal = env.step(a)
            s = s_
            if is_goal:
                goal_count += 1
            if j == SETTING.MAX_EP_STEPS - 1 or done == True:
                reward_sum += r
                step_count += j
                break
    goal_rate = round(goal_count/test_round*100, 2)
    print("===============================================================================")
    print("goal_rate:{}% average_reward:{}, step_count:{}"
              .format(goal_rate, reward_sum/test_round, step_count))
    return



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
    state_dim = SETTING.STATE_DIMENTION
    act_dim = SETTING.ACT_DIMENTION
    act_bound = SETTING.ACT_BOUND
    start_time = time.time()

    ddpg = DDPG(act_dim, state_dim, act_bound)
    # ddpg.initial_replay_memory(env)
    # ddpg.train(env)

    ddpg.restore_model("/home/ccchang/ddpg_refactor/saved_model/0424_first_step/0424_firstStep100.0.ckpt")
    ddpg.test(env, 10000)

    # ddpg.restore_model("/home/ccchang/ddpg_refactor/saved_model/0508_ddpgRRT_secondStep100.0.ckpt")
    # ddpg.test(env,10000)



    # input("stop")


    # ddpg1 = DDPG(act_dim, state_dim, act_bound)
    # ddpg1.restore_model("/home/ccchang/ddpg_refactor/saved_model/0424_first_step/0424_firstStep100.0.ckpt")
    # ddpg1.test(env, 1000)
    # ddpg2 = DDPG(act_dim, state_dim, act_bound)
    # ddpg2.restore_model("/home/ccchang/ddpg_refactor/saved_model/0424_second_step/0424_secondStep100.0.ckpt")
    # multistage_test(env, 1000, ddpg1, ddpg2)


    print('Running time: ', time.time() - start_time)



if __name__ == "__main__":
    main()











































