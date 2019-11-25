'''
generate best actions for the imitation learning
'''

import json
import os
import sys
import numpy as np
import math
import cv2 as cv



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
        self.most_action  = 3

        self.object_character = {
        'suck': [                                                     # the objects that are able to be suck
            'cube', 'bottle', 'book','calculator'
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
            actions_data = []    #start
            actions_mask = []
            ans = ques['answer']
            question = ques['question']
            action_times = 0
            if ques['type'] == 'exist_positive':  
                action_position = []
                action_times = 0
                ques_object = ques['obj'] 
                obj_order = self.my_env.ur5.object_type.index(ques['obj'])
                print(".............")
                print(ques_object)
                if ques_object not in self.object_character['cover']:
                    print("can not be covered")
                    actions_data.append([0,0,0,0,0])   #stop
                    actions_mask.append(1)
                    _,rgb_image_raw = self.my_env.camera.get_camera_data()
                    rgb_image = np.array(rgb_image_raw)
                    episode_rgb_images.append(rgb_image)
                else: 
                    action_position,act_type,act_name = self.is_targetobject_overlap(obj_order)
                    if act_type == 0:  #no covered
                        print("no cover")
                        actions_data.append([0,0,0,0,0])   #stop
                        actions_mask.append(1)
                        _,rgb_image_raw = self.my_env.camera.get_camera_data()
                        rgb_image = np.array(rgb_image_raw)
                        episode_rgb_images.append(rgb_image)
                    else:
                        while (act_type!=0) and (action_times<self.most_action):
                            print("%s cover"%(act_name))
                            _,rgb_image_raw = self.my_env.camera.get_camera_data()
                            rgb_image = np.array(rgb_image_raw)
                            self.my_env.UR5_action(action_position,act_type)
                            for i in range(4):
                                action_position[i] = action_position[i]/128
                            action_position.append(act_type)                          
                            actions_data.append(action_position)
                            actions_mask.append(1)                
                            episode_rgb_images.append(rgb_image)
                            action_position,act_type,act_name = self.is_targetobject_overlap(obj_order)
                            action_times += 1
                        actions_data.append([0,0,0,0,0])  #stop
                        actions_mask.append(1)
                        _,rgb_image_raw = self.my_env.camera.get_camera_data()
                        rgb_image = np.array(rgb_image_raw)
                        episode_rgb_images.append(rgb_image)

                   
               

            elif ques['type'] == 'exist_negative':
                print("...............")
                print("no action")
                actions_data.append([0,0,0,0,0])    #stop
                actions_mask.append(1)
                _,rgb_image_raw = self.my_env.camera.get_camera_data()
                rgb_image = np.array(rgb_image_raw)
                episode_rgb_images.append(rgb_image)
                action_times = 0

            while(len(actions_data)<4):
                actions_data.append([0,0,0,0,0])  #stop
                actions_mask.append(0)
                _,rgb_image_raw = self.my_env.camera.get_camera_data()
                rgb_image = np.array(rgb_image_raw)
                episode_rgb_images.append(rgb_image)

            if(len(episode_rgb_images)!=4):
                print("error")
 
            result = {
                        "actions": actions_data,
                        "mask": actions_mask,
                        "action_length":action_times,
                        "answer": ans,
                        "rgb_images": episode_rgb_images,
                        "question": question,
                }
        

            all_action.append(result)
   
        return all_action



    def is_targetobject_overlap(self,target_order):

        obj_dic = self.my_env.ur5.get_obj_positions_and_orientations()
        overlap_rate,overlap_order = self.my_env.ur5.check_overlap(target_order,obj_dic)   # return the cover handle
        ant_type = 0
        action_position = [0,0,0,0]
        action_obj_name =''

        if overlap_rate < 0.4:    #no cover
            ant_type =0
            action_position = [0,0,0,0]
        else:

            obj_cover_type = self.my_env.ur5.object_type[overlap_order]

            obj_cover_position = self.my_env.camera.world2pixel(obj_dic[overlap_order]['position'])

            target_position =  self.my_env.camera.world2pixel(obj_dic[target_order]['position'])

            if obj_cover_type  in self.object_character['suck']:
                ant_type = 2
                action_position = obj_cover_position + obj_cover_position
            else:
                ant_type = 1
                action_position = target_position+ obj_cover_position
        action_obj_name = obj_dic[overlap_order]['name']
        return action_position,ant_type,action_obj_name


        

       









        
                
                


                
            

