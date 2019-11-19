import enviroment
import time

my_robot = enviroment.UR5(is_testing=1)
my_robot.ur5suction([0.5, 0, 0.8])
#my_robot.ur5loose([0.5,0,0.8])
