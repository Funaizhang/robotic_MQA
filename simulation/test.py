import enviroment
import time
import numpy as np

order = input("order")
test_file_name = 'test-10-obj-'+str(order)+'.txt'
my_robot = enviroment.UR5(is_testing=0,testing_file=test_file_name)
my_robot.disconnect()



