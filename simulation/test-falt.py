'''
generate training data for the imitation learning
'''
import os
import enviroment
from generate_questions import Qusetion
from generate_actions import best_action
import h5py


scene = "test-10-obj-05.txt"
my_robot = enviroment.UR5(is_testing=1,testing_file=scene)
my_robot.get_obj_positions_and_orientations()
my_robot.get_obj_positions_and_orientations_1()



                    





        
                
                


                
            

