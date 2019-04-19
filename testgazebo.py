import robot_control_interface_gazebo
import time
import math
import os
import signal



class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True







if __name__ == "__main__":

    killer = GracefulKiller()


    #you need to set different gazebo server address before server and client start
    os.environ["GAZEBO_MASTER_URI"] = "http://localhost:11340"



    ###(not needed)Set the model path of gazebo so that the <uri> in xml can be set as relative path
    #print(os.path.dirname(os.path.abspath(__file__)))
    #modelpath = os.path.dirname(os.path.abspath(__file__))
    #modelpath = modelpath + "/model/"
    #os.environ['GAZEBO_MODEL_PATH'] = modelpath





    import numpy as np
    np.set_printoptions(precision=4, linewidth=100)


    

    #this string is just a string
    gazebo_robot = robot_control_interface_gazebo.robot_control_interface_gazebo("test") 

    #set to print debug information
    gazebo_robot.setdebug(1)

    #load a costomized .sdf file(.xml or .world)
    #
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #the path in the gazebo .sdf(.xml or .world) file in the tag MUST be an absolute path.
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #


    ###select a robot type
    ROBOTNAME = "vs060"
    #ROBOTNAME = "cobotta"
    gazebo_robot.GAZEBO_StartAndLoadWorldWithTimer("./robot_control_interface/model/" + ROBOTNAME + "_gazebo.xml", 1000, "default", ROBOTNAME + "::J3::J3_visual")
    #gazebo_robot.GAZEBO_StartAndLoadWorld("./robot_control_interface/model/" + ROBOTNAME + "_gazebo.xml")

    


    gazebo_robot.GAZEBO_InitAllSensors("./robot_control_interface/model/" + ROBOTNAME + "_gazebo.xml")


    


    #gazebo_robot.GAZEBO_StartAndLoadWorld("worlds/mud.world")
    #load a gazebo .world file in the gazebo directory which is set in some system PATH
    #gazebo_robot.GAZEBO_StartAndLoadWorld("worlds/shapes_bitmask.world")
    
    """
    #add a jacobian chain to calculate jacobian for a certain link
    #in gazebo .sdf files(.xml or .world) there is no <site> only <link>(the same as <body>)

    gazebo_robot.GAZEBO_AddJacobianChain(
        "./model/vs060_gazebo.xml", #file name as the same as in GAZEBO_StartAndLoadWorld()
        "firstjac",                 #the jacobian chain named by you
        ROBOTNAME,                    #model name in <model> tag
        "BASE_link",                #the base <link> name
        "J6",                       #the tip <link> name
        [                           #the joints needed to calculate the jacobian
        "joint1",                   #fixed joints are not included
        "joint2",                   #this should be the same as the function's output
        "joint3", 
        "joint4", 
        "joint5", 
        "joint6"],
        )
    #"""





    #timestep size
    stepnum = 8
    #time.sleep(5)
    fnum = 0
    for loopi in range(1000):
        if loopi == 500:
            gazebo_robot.GAZEBO_Reset()

        #dictionary for position, velocity and force(torque)
        xdict = {          
        }   
        vdict = {
            "joint1" : 0.0,               
            "joint2" : 0.0,
            "joint3" : 0.0,   
            "joint4" : 0.0,               
            "joint5" : 0.0,
            "joint6" : 0.0,   
        }                   
        udict = {
        }   


        #dictionary of joint positon to calculate jacobian
        jdict = {
            "joint1" : 0.2,               
            "joint2" : 0.5,
            "joint3" : 0.4,   
            "joint4" : 0.5,               
            "joint5" : 1.5,
            "joint6" : 0.5,   
        } 

        xdict = gazebo_robot.GAZEBO_GetPosition(ROBOTNAME)
        xdict["joint1"] = xdict["joint1"] + 0.08
        #set all things of all joints, if joints name is not included, it will not be set
        gazebo_robot.GAZEBO_SetModel(ROBOTNAME, xdict, vdict, udict)   
        

        #step some timesteps
        gazebo_robot.GAZEBO_Step(stepnum)


        #get link jacobian by pos in jdict
        #print(gazebo_robot.GAZEBO_GetLinkJacByNameByPos("firstjac", jdict))

        #get link jacobian by the joint position of now
        #print(gazebo_robot.GAZEBO_GetLinkJacByNameByNow("firstjac"))

        #get a link's global position and rotation [X, Y, Z, QW, QX, QY, QZ]
        #print(gazebo_robot.GAZEBO_GetLinkPQByName(ROBOTNAME, "J6"))

        #get a model's global position and rotation [X, Y, Z, QW, QX, QY, QZ]
        #print(gazebo_robot.GAZEBO_GetModelPQByName(ROBOTNAME))


        #j5pos7 = gazebo_robot.GAZEBO_GetLinkPQByName(ROBOTNAME, "BASE_link")
        #j5pos7["X"] = j5pos7["X"] + 0.00
        #j5pos7["Y"] = j5pos7["Y"] + 0.00
        #j5pos7["Z"] = j5pos7["Z"] - 0.01

        #set a link's global position and rotation [X, Y, Z, QW, QX, QY, QZ]
        #gazebo_robot.GAZEBO_SetLinkPQByName(ROBOTNAME, "BASE_link", j5pos7)



        #j5pos7 = gazebo_robot.GAZEBO_GetLinkRelativePQByName(ROBOTNAME, "J5")
        #j5pos7["X"] = j5pos7["X"] + 0.01
        #j5pos7["Y"] = j5pos7["Y"] + 0.01
        #j5pos7["Z"] = j5pos7["Z"] + 0.01

        #set a link's global position and rotation [X, Y, Z, QW, QX, QY, QZ]
        #gazebo_robot.GAZEBO_SetLinkRelativePQByName(ROBOTNAME, "J5", j5pos7)




        #vspos7 = gazebo_robot.GAZEBO_GetModelPQByName(ROBOTNAME)
        #vspos7["X"] = vspos7["X"] + 0.005
        #vspos7["Z"] = vspos7["Z"] + 0.005

        #set a model's global position and rotation [X, Y, Z, QW, QX, QY, QZ]
        #gazebo_robot.GAZEBO_SetModelPQByName(ROBOTNAME, vspos7)



        
        #get all joints' position
        #print(gazebo_robot.GAZEBO_GetPosition(ROBOTNAME))

        #get all joints' velocity
        #print(gazebo_robot.GAZEBO_GetVel(ROBOTNAME))

        #get a joint position
        #print(gazebo_robot.GAZEBO_GetPositionByName(ROBOTNAME, "joint1"))

        #get a joint velocity
        #print(gazebo_robot.GAZEBO_GetVelByName(ROBOTNAME, "joint4"))

        #get all contact name
        #and the parameter is not distance to show
        #it is the depth

        #rlist = gazebo_robot.GAZEBO_GetContactNames(0.00)
        #if len(rlist) == 0:
            #pass
        #else:
            #print(rlist)
            #print("----------------------------")


        #get the raw image by camara name
        #the raw image is save as a "char" type numpy array
        #with size of height * width * 3(RGB)

        #rdict = gazebo_robot.GAZEBO_GetRawImageByCameraName("camara_sensor")
        #print(rdict["height"], rdict["width"], rdict["imagesize"], rdict["imageformat"])
        #for ii in range(320 * 240 * 3):
            #if rdict["rawimage"][ii] != -78 and rdict["rawimage"][ii] != -27 and rdict["rawimage"][ii] != 0:
                #print(rdict["rawimage"][ii])
        
        #savefilename = "/home/liulanhai/rlwork/test/" + str(fnum) + ".jpg"
        #fnum = fnum + 1

        #save the image by camara name and file name
        #gazebo_robot.GAZEBO_SaveImageByCameraName("camara_sensor", savefilename)


    

        if killer.kill_now:
            break
            #if input('Terminate training (y/[n])? ') == 'y':
                #break
            #killer.kill_now = False


    #close a gazebo world
    gazebo_robot.GAZEBO_CloseWorld()
