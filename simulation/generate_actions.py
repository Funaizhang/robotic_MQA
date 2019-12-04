'''
generate best actions for the imitation learning
'''

import json
import os
import sys
import numpy as np
import math
import cv2 as cv
from scipy.interpolate import interp1d



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
            'cube',  'mouse', 'scissors', 'stapler','keyboard'
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
        initial_x = 0.0
        initial_y = 0.0 

        for ques in self.question_dic:
            robot_position = []
            actions_mask = []
            actions_data = []
            episode_rgb_images=[]
            actions_data = []    #start
            action_lengths = 0
            ans = ques['answer']
            question = ques['question']

            ques_object = ques['obj'] 
            print(".............")
            print(ques_object)

            if ques['type'] == 'exist_positive':
                obj_order = self.my_env.ur5.object_type.index(ques['obj'])  
                if ques_object not in self.object_character['cover']:
                    robot_position.append([initial_x,initial_y])   #start
                    _,rgb_image_raw = self.my_env.camera.get_camera_data()
                    rgb_image = np.array(rgb_image_raw)
                    episode_rgb_images.append(rgb_image)         #start image

                    print("can not be covered")
                    actions_data = [0,9]    #no action
                    actions_mask = [1,0]

                    robot_position.append([initial_x,initial_y])
                    _,rgb_image_raw = self.my_env.camera.get_camera_data()   #end image
                    rgb_image = np.array(rgb_image_raw)
                    episode_rgb_images.append(rgb_image)
                    action_lengths = 1
                else: 
                    action_position,act_type,act_name,act_data,mask,positions,act_length = self.is_targetobject_overlap(obj_order)
                    if act_type == 0:  #no covered
                        robot_position.append([initial_x,initial_y])   #start
                        _,rgb_image_raw = self.my_env.camera.get_camera_data()
                        rgb_image = np.array(rgb_image_raw)
                        episode_rgb_images.append(rgb_image)         #start image
                                        
                        print("no cover")
                        actions_data = [0,9]    #no action
                        actions_mask = [1,0]

                        robot_position.append([initial_x,initial_y]) 
                        _,rgb_image_raw = self.my_env.camera.get_camera_data()   #end image
                        rgb_image = np.array(rgb_image_raw)
                        episode_rgb_images.append(rgb_image)
                        action_lengths = 1
   

                    elif act_type == 1: #pushing
                            print("%s cover"%(act_name))
                            
                            actions_data = act_data
                            actions_mask = mask
                            robot_position = positions
                            action_lengths = act_length

                            _,rgb_image = self.my_env.camera.get_camera_data()   #image before pushing
                            for i in range(action_lengths):
                                episode_rgb_images.append(rgb_image) 

                            _,img_after = self.my_env.UR5_action(action_position,act_type)
                            for i in range(40-action_lengths):
                                episode_rgb_images.append(img_after)  


                    elif act_type == 2: #sucking
                            print("%s cover"%(act_name))

                            actions_data = act_data
                            actions_mask = mask
                            robot_position = positions
                            action_lengths = act_length
                                     
                            _,rgb_image = self.my_env.camera.get_camera_data()   #image before pushing
                            for i in range(action_lengths):
                                episode_rgb_images.append(rgb_image) 

                            img_after = self.my_env.UR5_action(action_position,act_type)
                            for i in range(40-action_lengths):
                                episode_rgb_images.append(img_after)

      

            elif ques['type'] == 'exist_negative':
                print("no action")
                robot_position.append([initial_x,initial_y])   #start
                _,rgb_image_raw = self.my_env.camera.get_camera_data()
                rgb_image = np.array(rgb_image_raw)
                episode_rgb_images.append(rgb_image)         #start image

                actions_data = [0,9]    #no action
                actions_mask = [1,0]

                robot_position.append([initial_x,initial_y])   #end
                _,rgb_image_raw = self.my_env.camera.get_camera_data()   #end image
                rgb_image = np.array(rgb_image_raw)
                episode_rgb_images.append(rgb_image)
                action_lengths = 1



            episode_rgb_images_shrink =[]
            while(len(actions_data)<40):
                actions_data.append(9)  #add mask
                actions_mask.append(0)
                _,rgb_image_raw = self.my_env.camera.get_camera_data()
                rgb_image = np.array(rgb_image_raw)
                episode_rgb_images.append(rgb_image)
                robot_position.append([initial_x,initial_y])

            for image in episode_rgb_images:
                shrink = cv.resize(image,(224,224),interpolation=cv.INTER_AREA)
                shrink = np.array(shrink)
                shrink = shrink.transpose((2,0,1))
                shrink = (shrink/255.0).astype(np.float16)
                episode_rgb_images_shrink.append(shrink)

 
            result = {
                        "actions": actions_data,
                        "mask": actions_mask,
                        "robot_positions":robot_position,
                        "action_lengths":action_lengths,
                        "answer": ans,
                        "rgb_images": episode_rgb_images_shrink,
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
        actions_data = []
        positions_data =[]
        actions_mask = []
        action_lengths = 0

        if overlap_rate < 0.05:    #no cover
            ant_type =0
            action_position = [0,0,0,0]   #STOP
        else:

            obj_cover_type = self.my_env.ur5.object_type[overlap_order]

            obj_cover_position = self.my_env.camera.world2pixel(obj_dic[overlap_order]['position'])

            target_position =  self.my_env.camera.world2pixel(obj_dic[target_order]['position'])


            if obj_cover_type  in self.object_character['suck']:
                ant_type = 2
                action_position = obj_cover_position + obj_cover_position
                position_before = [0,0]
                act_list,pos_list,act_length = self.caculate_dx(obj_cover_position,position_before)
                actions_data = [0]    #start 

                action_lengths = act_length
                actions_data = actions_data + act_list
                positions_data = pos_list

                actions_data.append(9) #end
                positions_data.append([0,0])

                add_num = 40 - len(actions_data)

                for i in range(len(actions_data)-1):   #create mask
                    actions_mask.append(1)
                actions_mask.append(0)

                for i in range(add_num):
                    actions_data.append(9)
                    actions_mask.append(0)
                    positions_data.append([0,0])

            else:
                ant_type = 1
                action_position = target_position+ obj_cover_position
                position_before = [0,0]
                act_list_1,pos_list_1,act_length_1 = self.caculate_dx(target_position,position_before)
                act_list_2,pos_list_2,act_length_2 = self.caculate_dx(obj_cover_position,target_position)
                actions_data = [0]    #start 
                actions_data = actions_data + act_list_1+[0]+act_list_2
                action_lengths = act_length_1 + act_length_2
                positions_data = pos_list_1 +pos_list_2

                actions_data.append(9) #end
                positions_data.append([0,0])

                add_num = 40 - len(actions_data)

                for i in range(len(actions_data)-1):   #create mask
                    actions_mask.append(1)
                actions_mask.append(0)
                
                if add_num>0:
                    for i in range(add_num):
                        actions_data.append(9)
                        actions_mask.append(0)
                        positions_data.append([0,0])
                else:
                    action_lengths = action_lengths+add_num
                    actions_data = actions_data[0:39]
                    actions_data.append(9)
                    actions_mask = actions_mask[0:39]
                    actions_mask.append(0)
                    positions_data = positions_data[0:39]
                    positions_data.append([0,0])
 
                              
        action_obj_name = obj_dic[overlap_order]['name']
        return action_position,ant_type,action_obj_name,actions_data,actions_mask,positions_data,action_lengths


    def caculate_dx(self,position,position_before):
        dx = []
        dy = []
        position_list = [position_before]
        step_normal = 2.56
        step_large  = 25.6
        x_distance = position[0]-position_before[0]
        y_distance = position[1]-position_before[1]
        x_curr = position_before[0]
        y_curr = position_before[1]
        

        large_x_times = int(abs(x_distance)/step_large)
        x_rest = abs(x_distance) - large_x_times*step_large
        normal_x_times = int(x_rest/step_normal)
        

        for i in range(large_x_times):
            if x_distance >0:
                dx.append(2)
                x_curr += step_large
            else:
                dx.append(4)
                x_curr -= step_large
            position_list.append([x_curr,y_curr])

        for i in range(normal_x_times):
            if x_distance>0:
                dx.append(1)
                x_curr += step_normal
            else:
                dx.append(3)
                x_curr -= step_normal
            position_list.append([x_curr,y_curr])

                
        large_y_times = int(abs(y_distance)/step_large)
        y_rest = abs(y_distance) - large_y_times*step_large
        normal_y_times = int(y_rest/step_normal)
        

        for i in range(large_y_times):
            if y_distance >0:
                dy.append(6)
                y_curr += step_large
            else:
                dy.append(8)
                y_curr -= step_large
            position_list.append([x_curr,y_curr])


        for i in range(normal_y_times):
            if y_distance>0:
                dy.append(5)
                y_curr += step_normal
            else:
                dy.append(7)
                y_curr -= step_normal
            position_list.append([x_curr,y_curr])

        all_act = dx + dy   
        action_lengths = large_x_times + normal_x_times + large_y_times + normal_y_times +1   
        return all_act,position_list,action_lengths
        


        

       









        
                
                


                
            

