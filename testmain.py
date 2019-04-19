
"""
testq.py        test for quaternion
testmulti.py    test for multi-thread
"""

"""        
robots = simulator.get_robot_list
robot = simulator.getRobot(ID or name)
robot.setangle(pose)
robot.get_tcp_position()
"""


"""
installation:
cd build
cmake ../src
make -j
cd ..
cd src/robot_control_interface
make

run test python file:
python testmain.py
"""



#if you want to implement your functions
#try to edit ./src/robot_control_interface/robot_control_interface.cpp
#you need
#1 add function in the class
#2 implement the c++ function
#3 export the function to boost_python in the buttom of the cpp file
#4 in folder ./src/robot_control_interface/ type make to compile a new .so file
#5 it will be generated in the ./ folder which has the testmain.py file

import robot_control_interface_mujoco
import time

print(robot_control_interface_mujoco.__file__)





if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(precision=4)

    #this string is just a string
    mujoco_simulator = robot_control_interface_mujoco.robot_control_interface_mujoco("test") 

    #test functions to test int, double and numpy arrays
    mujoco_simulator.testprint()
    retnum = mujoco_simulator.testnumpyarrayinout(np.array([-0.3, 0.5, 0.8, 0.9, 1.11]))
    print(retnum)
    retnum = mujoco_simulator.testintinout(10)
    print(retnum)
    retnum = mujoco_simulator.testdoubleinout(200.535)   
    print(retnum)
    

    plist = ["bbbbc", "bbbbd"]
    retnum = mujoco_simulator.testpythonlistinout(plist)   
    print(retnum)




    #activate with the licence
    mujoco_simulator.MJC_Activate("./robot_control_interface/src/mujoco200/mjkey.txt")
    
    #load a xml file
    mujoco_simulator.MJC_LoadWorld("./robot_control_interface/model/cobotta_3.xml")

    #create a window
    mujoco_simulator.MJC_InitViewer(1024, 768, 0, 0, 0, 0, 0, 0, 0)

    #the joint angle, the number must match the number of joints in the xml file
    initx = np.array([0.75, 1.0, 1.1, 0.0, 0.8, 0.5])

    #the joint velocity, the number must match the number of joints in the xml file
    initv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #the input force(torque), the number must match the number of motors in the xml file
    initu = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #set the simulation timestep 
    mujoco_simulator.MJC_SetTimestep(0.01)
    n = 0
    changeload = 0
    while True:
        n = n + 1
        if n > 10000:
            if changeload == 0:
                #load another xml file, if you want to, and do not want to quit
                #there is no way to reset the whole simulation
                #because reset the viewer causes seg fault in ubuntu
                mujoco_simulator.MJC_LoadWorld("./robot_control_interface/model/cobotta_2.xml")
                initx = np.array([0.0, 1.0, 1.1, 0.0, 0.8, 0.0])
                initv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                initu = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])

                changeload = 1



        #print "---------------------"
        #print initx
        #print initv
        #print initu
        #simulate 1 step
        mujoco_simulator.MJC_Step(initx, initv, initu)
        
        initx = mujoco_simulator.MJC_GetState()
        initv = mujoco_simulator.MJC_GetVel()

        initx[2] = initx[2] + 0.04
        continue
        #get the angle and velocity

        mujoco_simulator.MJC_Step_SetStateByName(0.0, 0.0, "j2x000")#set state by joint name
        mujoco_simulator.MJC_Step_SetStateByName(0.0, 0.0, "j3x000")#set state by joint name

        mujoco_simulator.MJC_Step_SetControlByName(0.0, "j4x000")#set control by joint name

        initx = mujoco_simulator.MJC_GetState()
        j0s = mujoco_simulator.MJC_GetStateByName("j1x000")#get angle by joint name
        print(j0s, initx)
        #print("initx:", initx)
        initv = mujoco_simulator.MJC_GetVel()
        j5v = mujoco_simulator.MJC_GetVelByName("j5x000")#get velocity by joint name
        print(j5v, initv)

        




        #if you want to control the robot, here you can change the angles of the 6 motor each timestep
        #initx = initx + [0.002, 0.0, 0.0, 0.0, 0.0, 0.0]
        #initv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        #get the xyz of a site
        s1 = mujoco_simulator.MJC_GetSiteXYZByName("site1")
        s2 = mujoco_simulator.MJC_GetSiteXYZByName("site2")
        s3 = mujoco_simulator.MJC_GetSiteXYZByName("site3")
        #print s1, s2, s3

        #get the jacobian of the site, size:6 x number of motors
        #the first 3 x number of motors is jacobian for xyz
        #the last 3 x number of motors is jacobian for rotation
        j1 = mujoco_simulator.MJC_GetSiteJacByName("site1")
        j2 = mujoco_simulator.MJC_GetSiteJacByName("site2")
        j3 = mujoco_simulator.MJC_GetSiteJacByName("site3")        
        #print j1, "\n", j2, "\n", j3, "\n"
        
        #get a body xyz by name defined in the xml
        bxyz = mujoco_simulator.MJC_GetBodyPositionByName("tcy")
        bxyz = bxyz + [0.002, 0.002, 0.002]
        #set a body xyz
        mujoco_simulator.MJC_SetBodyPositionByName("tcy", bxyz)
    


        #get a site rotation by quaternion
        squat = mujoco_simulator.MJC_GetSiteGlobalQuatByName("site1")
        spos = mujoco_simulator.MJC_GetSiteGlobalPosByName("site1")
        #print("squat:", squat, spos)
        bpos = mujoco_simulator.MJC_GetBodyGlobalPosByName("b6x000")
        bbquat = mujoco_simulator.MJC_GetBodyGlobalQuatByName("b6x000")
        #print("bbquat:", bbquat, bpos)
        #get a body rotation by quaternion




        bquat = mujoco_simulator.MJC_GetBodyQuatByName("tcy2")
        #if you want to rotate a body 
        #just change the quaternion here
        bquat = bquat + [0.01, 0.0, 0.0, 0.0]
        #set a body quaternion
        mujoco_simulator.MJC_SetBodyQuatByName("tcy2", bquat)  
    
        #get the contact list which are defined in the xml file
        #this means that get the contact pairs whose distance is less than 0.01
        #a contact pair list is like: [[name of geom1, name of geom2, distance], ...]
        contactlist = mujoco_simulator.MJC_GetContactNames(0.01)
        #if len(contactlist) > 0:
            #print contactlist

        velocity = mujoco_simulator.MJC_Quat2Vel(bquat, 1)
        print(velocity)
        
        mujoco_simulator.MJC_IntegratePos(np.array([0.1]*6), 1)
    #the window cannot be closed without a seg fault in linux 
