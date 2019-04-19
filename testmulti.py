import robot_control_interface
import time
import _thread
import numpy as np
def getangle(robot_control_interface):
    n=0
    while True:
        n=n+1
        time.sleep(1)
        initxx = mujoco_simulator.MJC_GetState()
        initxx = initxx + [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
        mujoco_simulator.MJC_Step_SetState(initxx, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        #mujoco_simulator.MJC_Step_SetControl(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        print("--------", initxx)
        if n>5:
            break


if __name__ == "__main__":
    
    np.set_printoptions(precision=4)

    #this string is just a string
    mujoco_simulator = robot_control_interface.robot_control_interface("test") 

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
    mujoco_simulator.MJC_Activate("./src/mujoco200/mjkey.txt")
    
    #load a xml file
    mujoco_simulator.MJC_LoadWorld("./model/cobotta_3.xml")

    #create a window
    mujoco_simulator.MJC_InitViewer(1024, 768, 0, 0, 0, 0, 0, 0, 0)

    #the joint angle, the number must match the number of joints in the xml file
    initx = np.array([0.75, 1.0, 1.1, 0.0, 0.8, 0.0])

    #the joint velocity, the number must match the number of joints in the xml file
    initv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #the input force(torque), the number must match the number of motors in the xml file
    initu = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #set the simulation timestep 
    mujoco_simulator.MJC_SetTimestep(0.01)
    n = 0
    changeload = 0

    mujoco_simulator.MJC_Step(initx, initv, initu)

    _thread.start_new_thread(getangle, (mujoco_simulator, ))

    while True:
        #print "---------------------"
        #print initx
        #print initv
        #print initu
        #simulate 1 step
        mujoco_simulator.MJC_Step_Simulate1()
        mujoco_simulator.MJC_Step_Simulate2()
        mujoco_simulator.MJC_Step_Render()        
        #mujoco_simulator.MJC_Step(initx, initv, initu)
        


        #get the angle and velocity
        initxx = mujoco_simulator.MJC_GetState()
        #print(initxx, initx)
        #initv = mujoco_simulator.MJC_GetVel()

        #if you want to control the robot, here you can change the angles of the 6 motor each timestep
        initx = initx + [0.002, 0.0, 0.0, 0.0, 0.0, 0.0]
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

    #the window cannot be closed without a seg fault in linux 