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
scene = "test-10-obj-41.txt"
my_enviroment = enviroment.Environment(is_testing=0,testing_file = scene)
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
robot_positions =[]
mask = []
answers =[]
action_lengths = []
for data in all_data:
    images.append(data['rgb_images'])
    questionTokens = my_question.tokenize(
            data['question'], punctToRemove=['?'], addStartToken=False)
    encoded_question = my_question.encode(questionTokens, vocab['questionTokenToIdx'])
    encoded_question.append(0)
    questions.append(encoded_question)

    encode_answer = vocab['answerTokenToIdx'][str(data['answer'])]
    answers.append(encode_answer)


    actions.append(data['actions'])
    mask.append(data['mask'])
    robot_positions.append(data['robot_positions'])
    action_lengths.append(data['action_lengths'])
    
    '''
    print(data['actions'])
    print(data['mask'])
    print(data['robot_positions'])
    print(len(data['rgb_images']))
    print(encoded_question)
    print("...........")
    '''
    if len(data['actions']) != 22:
        print('ASB')
    if len(data['mask']) != 22:
        print('MSB')
        print(len(data['mask']))
    if len(data['robot_positions']) !=22:
        print('RSB')
        print(len(data['robot_positions']))
    if len(data['rgb_images']) !=22:
        print('ISB')
        print(len(data['rgb_images']))

f  = h5py.File("scene41.h5",'w')
f['images'] = images
f['questions'] = questions
f['actions']  = actions
f['robot_positions'] = robot_positions
f['mask'] = mask
f['answers'] = answers
f['action_lengths'] = action_lengths
                    





        
                
                


                
            

