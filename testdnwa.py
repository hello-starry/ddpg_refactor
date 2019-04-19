import robot_control_interface_dnwa
import time
import math

ROBOTMOVE_PTP = 1                # Trajectory is curved line.
ROBOTMOVE_CP = 2                # Trajectory is straight line.

ROBOT_TYPE_COBOTTA = 1
ROBOT_TYPE_VS050 = 2
ROBOT_TYPE_VS060 = 3
ROBOT_TYPE_VS087 = 4

CAMERA_TYPE_CANON = 1

HAND_TYPE_COBOTTA = 1
HAND_TYPE_VS050 = 2


















###For test slavemode
SPEED = 10
TIMESTEP = 0.008
LENG1STEP = SPEED * TIMESTEP
SHAPES = [
    ["line", 150.0, 0.0, 150.0, -50.0],
    ["circle", 200.0, -50.0, 50.0, 1.0, 1.5],
    ["line", 200.0, -100.0, 225.0, -100.0],
    ["circle", 225.0, -75.0, 25.0, 1.5, 2.0],
    ["circle", 275.0, -75.0, 25.0, -1.0, -1.5],
    ["line", 275.0, -50.0, 300.0, -50.0],
]
def makepointlist(shapelist):
    snum = len(shapelist)

    remains = 0.0

    retlist = list()

    for si in range(snum):
        if shapelist[si][0] == "line":
            startpointx = shapelist[si][1]
            startpointy = shapelist[si][2]
            endpointx = shapelist[si][3]
            endpointy = shapelist[si][4]

            dx = endpointx - startpointx
            dy = endpointy - startpointy
            dl = math.sqrt(dx**2 + dy**2)

            if dl < remains:
                remains = remains - dl
                continue

            stepx = LENG1STEP * dx / dl
            stepy = LENG1STEP * dy / dl

            remainx = remains * dx / dl
            remainy = remains * dy / dl

            startpointx = startpointx + remainx
            startpointy = startpointy + remainy
            
            

            pointnum = int((dl - remains) / LENG1STEP)

            remains = LENG1STEP - ((dl - remains) - LENG1STEP * pointnum)

            for ppi in range(pointnum + 1):
                retlist.append([startpointx + ppi * stepx, startpointy + ppi * stepy])

            if si == snum - 1:
                retlist.append([endpointx, endpointy])

        elif shapelist[si][0] == "circle":
            rx = shapelist[si][1]
            ry = shapelist[si][2]
            rr = shapelist[si][3]
            starttheta = shapelist[si][4] * math.pi
            endtheta = shapelist[si][5] * math.pi

            if endtheta - starttheta >= 0:
                steptheta = LENG1STEP / rr
                remaintheta = remains / rr
            else:
                steptheta = -LENG1STEP / rr
                remaintheta = -remains / rr                

            starttheta = starttheta + remaintheta

            pointnum = int((endtheta - starttheta) / steptheta)

            remainstheta = (endtheta - starttheta) - steptheta * pointnum

            remains = LENG1STEP - abs(remainstheta * rr)


            for ppi in range(pointnum + 1):
                retlist.append([rx + rr * math.cos(starttheta + ppi * steptheta), ry + rr * math.sin(starttheta + ppi * steptheta)])

            if si == snum - 1:
                retlist.append([rx + rr * math.cos(endtheta), ry + rr * math.sin(endtheta)])

        else:
            raise shapeerror

    return retlist



















if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(precision=4)

    #this string is just a string
    dnwa_robot = robot_control_interface_dnwa.robot_control_interface_dnwa("test") 

    #set to print debug information
    dnwa_robot.setdebug(1)

    #Initiate a robot, either a wincapsIII simulator or a real robot will be OK
    #Input the IP and the robot type.
    dnwa_robot.DNWA_InitRobotWithIp(10, 3, 232, 36, ROBOT_TYPE_COBOTTA)

    #Set the robot speed, 0 means external speed, 50 means 50%
    dnwa_robot.DNWA_RobotSpeed(0, 50.0)

    #Move to a J(j-, j1, j2, j3, j4, j5)
    #And another parameter means the trajectory type
    dnwa_robot.DNWA_RobotMoveJ(np.array([90.0, 0.0, 30.0, 0.0, 30.0, 0.0]), ROBOTMOVE_PTP)

    #get the angle of 6 joints
    print(dnwa_robot.DNWA_GetAngle())

    #get the position of 8 numbers, P(x, y, z, rx, ry ,rz, the type of gesture, unused)
    print(dnwa_robot.DNWA_GetPos())

    #Move to a P(x, y, z + distance, rx, ry ,rz)
    #means move to 50.0 over the given position
    dnwa_robot.DNWA_Approach(np.array([0.0, 200.0, 100.0, 180.0, 0.0, 0.0, 261]), 50) 
    
    
    print(dnwa_robot.DNWA_GetAngle())
    print(dnwa_robot.DNWA_GetPos())



    #Move to a P(x, y, z, rx, ry ,rz, and the type of gesture, 261 means hand directed vertically to the ground)
    #And another parameter means the trajectory type
    dnwa_robot.DNWA_RobotMoveP(np.array([0.0, 200.0, 100.0, 180.0, 0.0, 0.0, 261]), ROBOTMOVE_PTP)
    print(dnwa_robot.DNWA_GetAngle())
    print(dnwa_robot.DNWA_GetPos())

    #means move to 50.0 over the current position
    dnwa_robot.DNWA_Depart(50)
    print(dnwa_robot.DNWA_GetAngle())
    print(dnwa_robot.DNWA_GetPos())
    

    #For test slavemode
    pointlist = makepointlist(SHAPES)
    nextpos = np.zeros((8))
    nextpos[7] = 0
    nextpos[6] = 261.0
    nextpos[5] = 180.0
    nextpos[4] = 0.0
    nextpos[3] = 180.0
    nextpos[2] = 135.0      
    nextpos[0:2] = pointlist[0][0:2]

    #move to a first place
    #the distance slavemode can move durining a time step is short
    #if the distance between the current position and the first position is too long
    #there will be an error
    dnwa_robot.DNWA_RobotMoveP(nextpos[0:7], ROBOTMOVE_PTP)


    #set the slave mode
    #0x201, 
    #2 means a 3 buffer mode, and the slavemove function returns when a buffer is empty
    #1 means a P mode
    #0x202 mean a J mode with 3 buffer
    #0x201 and 0x202 is often used
    dnwa_robot.DNWA_SetSlave(0x201)
    for ri in range(len(pointlist)):
        nextpos[0:2] = pointlist[ri][0:2]

        #tell the robot the next position
        #you do not need to count 8 ms here
        dnwa_robot.DNWA_SlaveMove(nextpos)


    #stop the slave mode
    dnwa_robot.StopSlave()



    #disconnect from the robot
    #need to do this or the real robot controller will get in some trouble with too many connections
    dnwa_robot.DNWA_DisconnectRobot()





#Other functions cannot be tested without a real machine
"""
    int32_t DNWA_InitHand(int handtype); 
    int32_t DNWA_InitCanonCameraWithIp(int ip1, int ip2, int ip3, int ip4, int cameratype); 

    int32_t DNWA_DisconnectHand();
    int32_t DNWA_DisconnectCamera();

	int32_t DNWA_ClearError();
	int32_t DNWA_MotorOn();
	int32_t DNWA_GiveTakeArm();

    int32_t DNWA_MoveHand(float distance);
    int32_t DNWA_StoreHandImg(const std::string& filename);

"""
