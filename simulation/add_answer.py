'''
generate training data for the imitation learning
'''
import os
import enviroment
from generate_questions import Qusetion
from generate_actions import best_action
import h5py
import copy





scene = "test-10-obj-00.txt"
h5file = "scene00.h5"
answers =[]
iamges =[]
questions =[]
actions = []
action_length = []
mask= []


f = h5py.File(h5file,"r")
images =f['images'] 
questions =f['questions'] 
actions =f['actions']  
action_length = f['action_length'] 
mask = f['mask'] 

images_1 =copy.deepcopy(images)
questions_1 = copy.deepcopy(questions)
actions_1 = copy.deepcopy(actions)
action_length_1 = copy.deepcopy(action_length)
mask_1 = copy.deepcopy(mask)

my_enviroment = enviroment.Environment(is_testing=1,testing_file = scene)
object_exist_list = my_enviroment.ur5.object_type
print("the objetct which is exist:")
print(object_exist_list)

my_question =Qusetion(object_exist_list)
all_ques =my_question.createQueue()
vocab = my_question.create_vocab()

answers =[]
for que in all_ques:
    encode_answer = vocab['answerTokenToIdx'][str(que['answer'])]
    answers.append(encode_answer)





f1  = h5py.File("scene01.h5",'w')
f1['answers'] = answers
f1['images'] = images_1
f1['questions'] = questions_1
f1['actions']  = actions_1
f1['action_length'] = action_length_1
f1['mask'] = mask_1

                    





        
                
                


                
            

