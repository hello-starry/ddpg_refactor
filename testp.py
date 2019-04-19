from robot_control_interface.env.gazebo_env.pullout_gazebo import pullout_gazebo

if __name__ == "__main__":
    teste = pullout_gazebo()
    for _ in range(1000):  
        #teste.step([0.025, 0.025, 0.025, 0.025, 0.025, 0.025])
        teste.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    teste.close()