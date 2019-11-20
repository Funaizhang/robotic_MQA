import enviroment
import time
import numpy as np

# my_robot = enviroment.UR5(is_testing=0)
# # my_robot.ur5suction([0.5, 0, 0.8])
# #my_robot.ur5loose([0.5,0,0.8])

# obj_dict = my_robot.get_obj_positions_and_orientations()
# target_obj = my_robot.object_handles[0]
# target_overlap_list = my_robot.check_overlap(target_obj, obj_dict)
# print(target_overlap_list)

my_list = [0, 1, 2]
test_list = np.asarray(my_list-1)
print(test_list)