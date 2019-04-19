import numpy as np
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z
def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)
def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]
def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = np.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v    
def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    return w, x, y, z
def q_to_axisangle(q):
    w, v = q[0], q[1:]
    theta = np.arccos(w) * 2.0
    return normalize(v), theta




#if you want to implement your functions
#try to edit ./src/robot_control_interface/robot_control_interface.cpp
#you need
#1 add function in the class
#2 implement the c++ function
#3 export the function to boost_python in the buttom of the cpp file
#4 in folder ./src/robot_control_interface/ type make to compile a new .so file
#5 it will be generated in the ./ folder which has the testmain.py file

import robot_control_interface
import time

if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(precision=4)

    #this string is just a string
    mujoco_simulator = robot_control_interface.robot_control_interface("test") 





    #activate with the licence
    mujoco_simulator.MJC_Activate("./src/mujoco200/mjkey.txt")
    
    #load a xml file
    mujoco_simulator.MJC_LoadWorld("./model/cobotta_3.xml")

    #create a window
    mujoco_simulator.MJC_InitViewer(1024, 768, 0, 0, 0, 0, 0, 0, 0)

    mujoco_simulator.MJC_SetTimestep(0.01)
    #the joint angle, the number must match the number of joints in the xml file
    initx = np.array([0.75, 1.0, 1.1, 0.0, 0.8])

    #the joint velocity, the number must match the number of joints in the xml file
    initv = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    #the input force(torque), the number must match the number of motors in the xml file
    initu = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    while True:

        #simulate 1 step
        mujoco_simulator.MJC_Step(initx, initv, initu)
        
        #get a body rotation by quaternion
        bquat = mujoco_simulator.MJC_GetBodyQuatByName("tcy2")
        #if you want to rotate a body 
        #just change the quaternion here


        q0 = (1.0, 0.0, 0.0, 0.0)
        x_axis_unit = (1.0, 0.0, 0.0)
        y_axis_unit = (0.0, 1.0, 0.0)
        z_axis_unit = (0.0, 0.0, 1.0)
        r1 = axisangle_to_q(x_axis_unit, 1.0)
        r2 = axisangle_to_q(y_axis_unit, 0.5)
        r3 = axisangle_to_q(z_axis_unit, 0.2)        

        

        v = q_mult(r3, q0)
        v = q_mult(r2, v)
        v = q_mult(r1, v)


        print(bquat, v)
        #bquat = bquat + [0.01, 0.0, 0.0, 0.0]
        #set a body quaternion
        mujoco_simulator.MJC_SetBodyQuatByName("tcy2", bquat)  
    
