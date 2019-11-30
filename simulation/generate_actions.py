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
        push_mask = []
        suck_mask = []
        initial_x = 0.0
        initial_y = 0.0 

        for i in range(21):     #get mask of pushing action and sucking action
            push_mask.append(1)
        push_mask.append(0)
        for i in range(11):
            suck_mask.append(1)
        for i in range(11):
            suck_mask.append(0)

        print(suck_mask)
        print(push_mask)

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
                    actions_data = [[1,1],[0,0]]    #no action
                    actions_mask = [1,0]

                    robot_position.append([initial_x,initial_y])
                    _,rgb_image_raw = self.my_env.camera.get_camera_data()   #end image
                    rgb_image = np.array(rgb_image_raw)
                    episode_rgb_images.append(rgb_image)
                    action_lengths = 1
                else: 
                    action_position,act_type,act_name,act_data = self.is_targetobject_overlap(obj_order)
                    if act_type == 0:  #no covered
                        robot_position.append([initial_x,initial_y])   #start
                        _,rgb_image_raw = self.my_env.camera.get_camera_data()
                        rgb_image = np.array(rgb_image_raw)
                        episode_rgb_images.append(rgb_image)         #start image
                                        
                        print("no cover")
                        actions_data = [[1,1],[0,0]]    #no action
                        actions_mask = [1,0]

                        robot_position.append([initial_x,initial_y]) 
                        _,rgb_image_raw = self.my_env.camera.get_camera_data()   #end image
                        rgb_image = np.array(rgb_image_raw)
                        episode_rgb_images.append(rgb_image)
                        action_lengths = 1
   

                    elif act_type == 1: #pushing
                            print("%s cover"%(act_name))

                            
                            actions_data = act_data
                            actions_mask = push_mask

                            dx = act_data[1][0]*25.6
                            dy = act_data[1][1]*25.6

                            _,rgb_image_raw = self.my_env.camera.get_camera_data()   #image before pushing
                            rgb_image = np.array(rgb_image_raw)

                            for i in range(11):
                                episode_rgb_images.append(rgb_image)
                                robot_position.append([(0+dx*i)/256.0,(0+dy*i)/256.0])    #start and little moving before pushing

                            img_before,img_after = self.my_env.UR5_action(action_position,act_type)

                            dx1 = act_data[12][0]*2.56
                            dy1 = act_data[12][1]*2.56
                            target_x = action_position[0]
                            target_y = action_position[1]
                            cover_x = action_position[2]
                            cover_y = action_position[3]

                            for i in range(10):
                                episode_rgb_images.append(img_before)
                                robot_position.append([(target_x+dx1*(i+1))/256.0,(target_y+dy1*(i+1))/256.0])    #little moving during pushing

                            episode_rgb_images.append(img_after)   #after pushing

                            robot_position.append([cover_x/256.0,cover_y/256.0])  
                            print(robot_position[10],robot_position[20])
                            action_lengths = 21

                    elif act_type == 2: #sucking
                            print("%s cover"%(act_name))
                           
                            actions_data = act_data
                            actions_mask = suck_mask

                            dx = act_data[1][0]*25.6
                            dy = act_data[1][1]*25.6

                            _,rgb_image_raw = self.my_env.camera.get_camera_data()   #image before sucking
                            rgb_image = np.array(rgb_image_raw)

                            for i in range(11):
                                episode_rgb_images.append(rgb_image)
                                robot_position.append([(0+dx*i)/256.0,(0+dy*i)/256.0])    #little moving before sucking and start

                            img_after = self.my_env.UR5_action(action_position,act_type)

                            target_x = action_position[0]
                            target_y = action_position[1]

                            for i in range(11):
                                episode_rgb_images.append(img_after)
                                robot_position.append([target_x/256.0,target_y/256.])
                            print(robot_position[10])
                            action_lengths = 11
                   
                            
        

            elif ques['type'] == 'exist_negative':
                print("no action")
                robot_position.append([initial_x,initial_y])   #start
                _,rgb_image_raw = self.my_env.camera.get_camera_data()
                rgb_image = np.array(rgb_image_raw)
                episode_rgb_images.append(rgb_image)         #start image

                actions_data = [[1,1],[0,0]]    #no action
                actions_mask = [1,0]

                robot_position.append([initial_x,initial_y])   #end
                _,rgb_image_raw = self.my_env.camera.get_camera_data()   #end image
                rgb_image = np.array(rgb_image_raw)
                episode_rgb_images.append(rgb_image)
                action_lengths = 1



            episode_rgb_images_shrink =[]
            while(len(actions_data)<22):
                actions_data.append([0,0])  #add mask
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
                dx,dy = self.caculate_dx(obj_cover_position,position_before)
                actions_data = [[1,1]]    #start 
                for i in range(10):
                    actions_data.append([dx,dy])
                for i in range(10):
                    actions_data.append([0,0])
                actions_data.append([0,0])  #end

            else:
                ant_type = 1
                action_position = target_position+ obj_cover_position
                position_before = [0,0]
                dx,dy = self.caculate_dx(target_position,position_before)
                dx_1,dy_1 = self.caculate_dx(obj_cover_position,target_position,2.56)
                actions_data = [[1,1]]  #start
                for i in range(10):
                    actions_data.append([dx,dy])
                for i in range(10):
                    actions_data.append([dx_1,dy_1])
                actions_data.append([0,0]) #end
                
                
        action_obj_name = obj_dic[overlap_order]['name']
        return action_position,ant_type,action_obj_name,actions_data


    def caculate_dx(self,position,position_before,nomal_ratio = 25.6):
        x = position[0]
        y = position[1]
        x_before = position_before[0]
        y_before = position_before[1]
        step_size = 10.0
        nomal_ratio = nomal_ratio
        dx = (x-x_before)/step_size
        dy = (y-y_before)/step_size
        dx = dx/nomal_ratio
        dy = dy/nomal_ratio
        return dx,dy
        


        

       









        
                
                


                
            

