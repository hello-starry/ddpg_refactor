import random
import math
import copy
import sys
import time
import numpy as np
from env.gazebo_env.suction_grasp_env import *
from setting import SETTING

state_dim = SETTING.STATE_DIMENTION
act_dim = SETTING.ACT_DIMENTION
act_bound = SETTING.ACT_BOUND

class Node():
    """
    RRT Node
    """
    def __init__(self, joint_pos, parent=None, next=None):
        self.joint_pos = np.array(joint_pos)
        self.parent = parent
        self.next = next

class RRT():
    u"""
    Class for RRT Planning
    """
    def __init__(self, env, start, goal, goal_sample_rate, expand_factor, max_iteration):
        u"""
        Setting Parameter

        start:Start Position [J1,J2...J6]
        goal:Goal Position [J1,J2...J6]
        """
        self.env = env
        self.act_dim = 6
        self.start_node = Node(start)
        self.end_node = Node(goal)
        self.goal_sample_rate = goal_sample_rate
        self.pos_limit = [[-0.5*np.pi,1.5*np.pi],[-0.75*np.pi,0.75*np.pi],[-0.75*np.pi,0.75*np.pi],
                          [-0.75 * np.pi, 0.75 * np.pi],[-0.75*np.pi,0.75*np.pi],[-0.75*np.pi,0.75*np.pi]]

        # self.pos_limit = [[1.5 * np.pi, 1.7 * np.pi], [-0.75 * np.pi, 0.1 * np.pi], [-0.75 * np.pi, 0.75 * np.pi],
        #                   [-0.75 * np.pi, 0.75 * np.pi], [-0.25 * np.pi, 0.25 * np.pi], [-0.1, 0.1]]

        # self.pos_limit = [[-np.pi*0.75, np.pi*0.75],[-np.pi*0.75, np.pi*0.75],[-np.pi*0.75, np.pi*0.75],
        #                  [-np.pi*0.75, np.pi*0.75],[-np.pi*0.75, np.pi*0.75],[-np.pi*0.75, np.pi*0.75],]

        self.act_limit = [-0.1,0.1]
        self.expand_factor = expand_factor
        self.max_iteration = max_iteration
        self.min_distance = 1000000

    def plan(self, animation=True):
        """
        Pathplanning

        animation: flag for animation on or off
        """

        self.node_list = [self.start_node]
        is_goal = False
        iteration = 0

        while not is_goal and iteration < self.max_iteration:
            # print("Iteration:",iteration)
            if np.random.rand() > self.goal_sample_rate:
                sample_pos = np.zeros(self.act_dim,)
                for i in range(len(self.pos_limit)):
                    sample_pos[i] = np.random.uniform(low=self.pos_limit[i][0], high=self.pos_limit[i][1])
            else:
                sample_pos = self.end_node.joint_pos

            # sample_pos = self.end_node.joint_pos

            # Find nearest node
            nearest_idx = self.GetNearestListIndex(self.node_list, sample_pos)
            # expand tree
            nearest_node = self.node_list[nearest_idx]
            # print("nearest_idx:",nearest_idx)
            # print("nearest_node_pos:",self.node_list[nearest_idx])
            # input("stop1")

            diff = sample_pos - nearest_node.joint_pos
            e_factor = self.expand_factor
            while e_factor >= 0.1:
                # if abs(np.max(diff)) >= self.act_limit:
                #     break
                # print("nearest___:",nearest_node.joint_pos)
                # print("sample_pos:", sample_pos)
                # print("diff:",diff)
                # print("expand_factor:",self.expand_factor)
                act = np.clip(diff*e_factor,self.act_limit[0],self.act_limit[1])
                new_pos = nearest_node.joint_pos + act
                # print("new_pos:", new_pos)
                # input("stop")
                e_factor /= 2.
                if not self.is_collide(new_pos):
                    new_node = Node(joint_pos=new_pos, parent=nearest_node)
                    self.node_list.append(new_node)
                    # print("Expand success")
                    # check goal
                    diff_list = abs(self.end_node.joint_pos - new_node.joint_pos)
                    # print("new node:", new_node.joint_pos, "diff_list:",diff_list)
                    if (diff_list <= 0.05).all():
                        print("Goal!!")
                        is_goal = True
                        # Construct the solution path
                        while new_node.parent != None:
                            new_node.parent.next = new_node
                            new_node = new_node.parent
                        print("iteration:", iteration)
                        return self.start_node
                    break
            # g_idx = self.GetNearestListIndex(self.node_list, self.end_node.joint_pos)
            # min_distance = np.sum(np.sqrt(np.square(self.node_list[g_idx].joint_pos - self.end_node.joint_pos)))
            # if min_distance < self.min_distance:
            #     self.min_distance = min_distance
            #     print("min distance:",min_distance)
            iteration += 1

        return None

    def is_collide(self, pos):
        self.env.interface.GAZEBO_SetModel("vs060",make_6joint_dict(pos),{},{})
        collide_list = self.env.interface.GAZEBO_GetContactNames(0.00)
        if len(collide_list) != 0:
            return True
        else:
            return False

    def GetNearestListIndex(self, node_list, sample_pos):
        distance_list =[np.sum(np.square(node.joint_pos-sample_pos)) for node in node_list]
        min_idx = distance_list.index(min(distance_list))
        return min_idx


def rrt_step1_test(env):
    goal_count = 0
    # start_pos = [4.61, -0.92, -0.46, 1.0, -1.54, 0.]
    start_pos = [4.21, -0.92, -0.26, 0.8, -1.34, 0.1]
    end_pos = [4.71, -1.12, -0.46, 0., -1.54, 0.]
    max_iteration = 1000
    g_rate = 0.05
    round = 20

    for i in range(round):
        env.reset()
        env.interface.GAZEBO_SetModel("vs060", make_6joint_dict(start_pos), {}, {})
        rrt = RRT(env=env, start=start_pos, goal=end_pos, goal_sample_rate=g_rate, max_iteration=max_iteration)
        ptr = rrt.plan(animation=False)
        if not ptr:
            continue
        else:
            goal_count += 1
            while ptr != None:
                print(ptr.joint_pos)
                env.interface.GAZEBO_SetModel("vs060",make_6joint_dict(ptr.joint_pos),{},{})
                env.interface.GAZEBO_Step(1)
                # collide_list = env.interface.GAZEBO_GetContactNames(0.05)
                # if len(collide_list) != 0:
                #     env.interface.GAZEBO_AddJoint("vs060", "J6", "panel", "panel_link")
                time.sleep(0.04)
                ptr = ptr.next
            env.interface.GAZEBO_AddJoint("vs060", "J6", "panel", "panel_link")
            env.is_sucked = True

    print("goal_rate:", goal_count/round*100,"%")
    input("end")


def main():
    np.set_printoptions(precision=3)
    portStr = None
    if len(sys.argv) > 1:
        portStr = sys.argv[1]

    env = suction_grasp_env(portStr)
    # rrt_step1_test(env)

    tmp_pos = [1.57, 0.96, 0.828, 0., 1.36, 0.]
    env.interface.GAZEBO_SetModel("vs060", make_6joint_dict(tmp_pos), {}, {})
    env.interface.GAZEBO_Step(1)
    env.interface.GAZEBO_AddJoint("vs060", "J6", "panel", "panel_link")
    env.is_sucked = True
    input("stop")
    max_iteration = 1000
    g_rate = 0.05

    # Adjust end position manually
    # tmp_pos = [0.87, 0.91, 0.85, 0.80, 0.94, 0.] # step1_rrt_start
    tmp_pos = [1.71, -0.71, -0.87, -0.15, -1.76, -0.413] #step2_rrt_start
    tmp_pos = [4.68, 1.024, 0.39, -0.006, 1.5, 0]
    while True:
        env.interface.GAZEBO_SetModel("vs060",make_6joint_dict(tmp_pos),{},{})
        # env.interface.GAZEBO_SetModel("vs060",make_6joint_dict(end_pos2),{},{})
        # collide_list = env.interface.GAZEBO_GetContactNames(0.00)
        # if len(collide_list) != 0:
        #     print(collide_list)
        joint_dict = env.interface.GAZEBO_GetPosition("vs060")
        print("current joint:",joint_dict_to_list(joint_dict))
        tcp_pq = pq_dict_to_list(env.interface.GAZEBO_GetLinkPQByName("vs060","J6"))
        panel_pq = pq_dict_to_list(env.interface.GAZEBO_GetLinkPQByName("panel","panel_link"))
        tcp_q = tcp_pq[3:]
        panel_q = panel_pq[3:]
        orientation = 1-np.square(tcp_q[0]*panel_q[0]+tcp_q[1]*panel_q[1]+tcp_q[2]*panel_q[2]+tcp_q[3]*panel_q[3])
        print("oreintation:",orientation)
        env.interface.GAZEBO_Step(1)
        collide_list = env.interface.GAZEBO_GetContactNames(0)
        if len(collide_list) != 0:
            print(collide_list)
        str = input("input the moving value:")
        str = str.split()
        if len(str)!=6:
            print("wrong input")
            continue
        for i in range(len(tmp_pos)):
            tmp_pos[i] += float(str[i])

    # input("start")
    while True:
        env.interface.GAZEBO_SetModel("vs060", make_6joint_dict(start_pos), {}, {})
        env.interface.GAZEBO_Step(1)
        rrt = RRT(env=env, start=start_pos, goal=end_pos, goal_sample_rate=g_rate, max_iteration=max_iteration)
        ptr = rrt.plan(animation=False)
        if not ptr:
            continue
        else:
            while ptr != None:
                print(ptr.joint_pos)
                env.interface.GAZEBO_SetModel("vs060",make_6joint_dict(ptr.joint_pos),{},{})
                env.interface.GAZEBO_Step(1)
                # collide_list = env.interface.GAZEBO_GetContactNames(0.05)
                # if len(collide_list) != 0:
                #     env.interface.GAZEBO_AddJoint("vs060", "J6", "panel", "panel_link")
                time.sleep(0.04)
                ptr = ptr.next
            env.interface.GAZEBO_AddJoint("vs060", "J6", "panel", "panel_link")
            env.is_sucked = True
            break

    current_pos = joint_dict_to_list(env.interface.GAZEBO_GetPosition("vs060"))
    while True:
        rrt = RRT(env=env, start=current_pos, goal=end_pos2, goal_sample_rate=g_rate, max_iteration=max_iteration)
        ptr = rrt.plan(animation=False)
        if not ptr:
            continue
        else:
            while ptr != None:
                print(ptr.joint_pos)
                env.interface.GAZEBO_SetModel("vs060",make_6joint_dict(ptr.joint_pos),{},{})
                env.interface.GAZEBO_Step(8)
                # collide_list = env.interface.GAZEBO_GetContactNames(0.05)
                # if len(collide_list) != 0:
                #     env.interface.GAZEBO_AddJoint("vs060", "J6", "panel", "panel_link")
                time.sleep(0.04)
                ptr = ptr.next
            env.interface.GAZEBO_RemoveJoint("vs060", "suction")
            break
    input("End")
    env.interface.GAZEBO_CloseWorld()




if __name__ == '__main__':
    main()
