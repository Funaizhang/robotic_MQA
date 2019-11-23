'''
generate best actions for the imitation learning
'''

import json
import os
import sys
import numpy as np
import math



class best_action():  
#init with the current scene    
    def __init__(self,env):
        self.my_env = env
        self.best_action_dir = os.path.abspath('../actions/action.json')
        self.question_dir = os.path.abspath('../questions/question.json')
        self.question_file = open(self.question_dir,encoding='utf-8')
        self.question_dic = json.load(self.question_file)
        self.question_file.close()
        self.obj_dict = self.my_env.ur5.get_obj_positions_and_orientations
        self.obj_type_exist = self.my_env.ur5.object_type   # the object which exists in the scene
        self.most_action  =3

        self.object_character = {
        'suck': [                                                     # the objects that are able to be suck
            'cube', 'bottle', 'book'
        ],
        'push': [                                                    # the objects that are able to be push
            'book', 'bottle', 'cup', 'calculator',
            'cube', 'keyboard', 'mouse', 'scissors', 'stapler','pc'
        ],                                                         # the objects that are likely to be covered
        'cover':[
             'bottle', 'cup', 'calculator','key','pen',
            'cube',  'mouse', 'scissors', 'stapler'
        ]       
        }

    def generate_best_action(self):   #
        '''
        an action is composed of two parts, the first parts is the type of the action,
        1:push 2:suck 3:no action
        the second part is the position of the part  
        [start_point_x,start_point_y,end_point_x,end_point_y]
        the start_point and end_point is same to the action of suckinng and loosing
        the start_point and end_point are both [0,0] to the action of no action
        '''
        obj_order = 0
        all_action  = []    #datas of all quesions in a scene
        for ques in self.question_dic:
            episode_rgb_images=[]
            actions_type= []
            actions = []
            ans = ques['answer']
            question = ques['question']
            if ques['type'] == 'exist_positive':  
                action_position = []
                action_times = 0
                ques_object = ques['obj'] 
                print(".............")
                print(ques_object)
                if ques_object not in self.object_character['cover']:
                    print("can not be covered")
                    actions_type.append(3)        
                    actions.append([3,3,3,3,3,3])     #no action
                    rgb_image,_ = self.my_env.camera.get_camera_data()
                    episode_rgb_images.append(rgb_image)
                else: 
                    action_position,act_type = self.is_targetobject_overlap(obj_order)
                    if act_type == 3:  #no covered
                        print("no cover")
                        actions_type.append(3)        
                        actions.append([3,3,3,3,3,3])    #no action
                        rgb_image,_ = self.my_env.camera.get_camera_data()
                        episode_rgb_images.append(rgb_image)
                    while (act_type!=3) and (action_times<self.most_action):
                        print("cover")     
                        actions_type.append(act_type)
                        actions.append(action_position)
                        rgb_image,depth_image = self.my_env.camera.get_camera_data()
                        self.my_env.action(action_position,act_type)                
                        episode_rgb_images.append(rgb_image)
                        action_position,act_type = self.is_targetobject_overlap(obj_order)
                        action_times += 1
                print("........")

            elif ques['type'] == 'exist_negative':
                print("...............")
                print("no action")
                actions_type.append(3)        
                actions.append([3,3,3,3,3,3])     #no action
                rgb_image,depth_image = self.my_env.camera.get_camera_data()
                episode_rgb_images.append(rgb_image)
                print("..............")

            result = {
                        "act_type": actions_type,
                        "actions": actions,
                        "answer": ans,
                        "rgb_images": episode_rgb_images,
                        "question": question,
                }

            all_action.append(result)
            obj_order +=1
        return all_action



    def is_targetobject_overlap(self,target_order):

        obj_dic = self.my_env.ur5.get_obj_positions_and_orientations()
        overlap_rate,overlap_order = self.my_env.ur5.check_overlap(target_order,obj_dic)   # return the cover handle
        ant_type = 3
        action_position = [3,3,3,3,3,3]

        if overlap_rate < 0.5:    #no cover
            ant_type =3
            action_position = [3,3,3,3,3,3]
        else:

            obj_cover_type = self.my_env.ur5.object_type[overlap_order]

            obj_cover_position = obj_dic[overlap_order]['position']

            target_position = obj_dic[target_order]['position']

            if obj_cover_type  in self.object_character['suck']:
                ant_type = 2
                action_position = obj_cover_position + obj_cover_position
            else:
                ant_type = 1
                action_position = target_position+ obj_cover_position
        return action_position,ant_type

       









        
                
                


                
            

