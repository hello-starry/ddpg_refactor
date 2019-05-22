


class SETTING(object):

    RESET_FROM_LOG = False
    RESET_FROM_RRT = False
    SAVE_MODEL = True
    SAVE_STATE = False

    MAX_EPISODES = 100000000
    MAX_EP_STEPS = 600
    NEURONNUM = 1000
    LR_A = 1e-5  # learning rate for actor
    LR_C = 1e-5  # learning rate for critic
    GAMMA = 0.9  # reward discount
    TAU = 0.001  # soft replacement
    MEMORY_CAPACITY = 20000
    BATCH_SIZE = 128

    IMG_STATE = False
    IMG_WIDTH = 0
    IMG_HEIGHT = 0
    IMG_CHANNEL = 0
    IMG_STATE_RANGE = (0.,1.)
    IMG_FRAME = 1

    STATE_DIMENTION = 7 + 6 + 7 + 7 + 1      #initial_tcp_pq + joint_angle + tcp_pq + goal_pq + is_sucked
    STATE_RANGE = (0.,1.)

    ACT_DIMENTION = 6
    ACT_BOUND = 0.05

    # self.infodict["obs_array"] = True
    # self.infodict["obs_array_size"] = 6 + 7 + 7 + 1  # Joint 6 angle + tcp pq + panel pq + is_sucked
    # self.infodict["obs_array_range"] = (0., 1.)
    # self.infodict["obs_pic"] = False
    # self.infodict["obs_pic_width"] = 0
    # self.infodict["obs_pic_height"] = 0
    # self.infodict["obs_pic_channel"] = 0
    # self.infodict["obs_pic_range"] = (0., 1.)
    # self.infodict["obs_frame"] = 1
    # self.infodict["act_array_type"] = "continuous"
    # self.infodict["act_array_size"] = 6
    # self.infodict["act_array_range"] = (0., 0.05)