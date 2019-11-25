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

my_question =Qusetion(object_exist_list)
my_question.createQueue()
vocab = my_question.create_vocab()

my_action = best_action(my_enviroment)
all_data = my_action.generate_best_action()
images = []
questions =[]
actions = []
action_length =[]
mask = []
for data in all_data:
    images.append(data['rgb_images'])
    questionTokens = my_question.tokenize(
            data['question'], punctToRemove=['?'], addStartToken=False)
    encoded_question = my_question.encode(questionTokens, vocab['questionTokenToIdx'])
    questions.append(encoded_question)
    actions.append(data['actions'])
    mask.append(data['mask'])
    print(data['actions'])
    print(data['mask'])
    print(encoded_question)
    print("...........")
    action_length.append(data['action_length'])

f  = h5py.File("scene1.h5",'w')
f['images'] = images
f['questions'] = questions
f['actions']  = actions
f['action_length'] = action_length
f['mask'] = mask
                    





        
                
                


                
            

