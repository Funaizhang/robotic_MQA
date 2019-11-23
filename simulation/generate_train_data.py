'''
generate training data for the imitation learning
'''
import os
import enviroment
from generate_questions import Qusetion
from generate_actions import best_action
import h5py
'''
scene_list_dir =os.path.abspath("simulation/test_cases")
scene_list = os.listdir(scene_list_dir)
for scene in scene_list:
    my_enviroment = simulation.enviroment.Environment(1,testing_file = scene)
    object_exist_list = my_enviroment.ur5.object_type
    print("the objetct which is exist:")
    print(object_exist_list)
    my_question =Qusetion(object_exist_list)
    my_action = best_action(my_enviroment)
'''
scene = "test-10-obj-01.txt"
my_enviroment = enviroment.Environment(is_testing=1,testing_file = scene)
object_exist_list = my_enviroment.ur5.object_type
print("the objetct which is exist:")
print(object_exist_list)
'''
my_question =Qusetion(object_exist_list)
my_question.createQueue()
'''
my_action = best_action(my_enviroment)
all_data = my_action.generate_best_action()

f = h5py.File('dataset.h5','w')   
f['data'] = imgData                
f['labels'] = range(100)           
f.close()                         





        
                
                


                
            

