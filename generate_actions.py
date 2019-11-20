'''
generate best actions for the imitation learning
'''

import json
import os
import pickle
import sys
import numpy as np
import math



class best_action():  
#init with the current scene    
    def __init__(self,env):
        self.my_env = env
        self.best_action_dir = os.path.abspath('actions/action.pk')
        self.question_dir = os.path.abspath('questions/question.json')
        question_file = open(self.question_dir,encoding='utf-8')
        self.question_dic = json.load(self.question_file)
        question_file.close()
        self.obj_dict = self.env.ur5.get_obj_positions_and_orientations
        self.obj_position,self.obj_oritentation = !!!!!
        self.obj_type_exist = self.env.ur5.object_type   # the object which exists in the scene
        self.most_action  =5

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
        0:no action 1:push 2:suck 3:loose
        the second part is the position of the part  
        [start_point_x,start_point_y,end_point_x,end_point_y]
        the start_point and end_point is same to the action of suckinng and loosing
        the start_point and end_point are both [0,0] to the action of no action
        '''
        obj_order = 0
        all_action  = []    #datas of all quesions in a scene
        for ques in self.question_dic:
            episode_rgb_images=[]
            episode_depth_image=[]
            actions_type= []
            actions = []
            ans = ques['answer']
            question = ques['question']

            if ques['type'] is 'exist_positive':  
                action_position = []
                action_times = 0
                ques_object = question['obj']
                if question_object not in self.object_character['cover']:
                    actions_type.append(0)        
                    actions.append([0,0,0,0])     #no action
                    rgb_image,depth_image = self.env.camera.get_camera_data()
                    eposide_rgb_images.append(rgb_image)
                    eposide_depth_images.append(depth_image)
                else: 
                    action_position,act_type = self.is_targetobject_covered(obj_order)
                    if act_type == 0:  #no covered
                        actions_type.append(0)        
                        actions.append([0,0,0,0])     #no action
                        rgb_image,depth_image = self.env.camera.get_camera_data()
                        eposide_rgb_images.append(rgb_image)
                        eposide_depth_images.append(depth_image)
                    while (act_type!=0) and (action_times<self.most_action):     
                        actions_type.append(act_type)
                        actions.append(action_position)
                        self.env.action(action_position)                
                        rgb_image,depth_image = self.env.camera.get_camera_data()
                        eposide_rgb_images.append(rgb_image)
                        eposide_depth_images.append(depth_image)
                        action_position,act_type = self.is_targetobject_covered(obj_order)
                        action_times += 1

            elif ques['type'] is 'exist_negative':
                    actions_type.append(0)        
                    actions.append([0,0,0,0])     #no action
                    rgb_image,depth_image = self.env.camera.get_camera_data()
                    eposide_rgb_images.append(rgb_image)
                    eposide_depth_images.append(depth_image)

            result = {
                        "act_type": actions_type,
                        "actions": actions,
                        "answer": ans,
                        "rgb_images": episode_rgb_images,
                        "depth_images": episode_depth_images,
                        "question": question,
                    }
            all_action.append(result)
        with open(self.shortest_path_dir, "wb") as f:
            pickle.dump(all_action, f)




 '''   
    def findclutter(depth_image):
        #crop to 64*64
        img_crop,img_location = self.most_clutter_area(depth_image) 
        #crop to 8*8
        img_crop_1,img_location_1,clutter_value = self.most_clutter_area(img_crop_1)

        clutter_postion = [img_location[0]*64 + img_location_1[0]*8 +4 , img_location[0]*64 + img_location_1[0]*8]

        return clutter_position,clutter_value

    def most_clutter_area(self,depth_image):   
        # the size of image should be 512*512
        img = np.array(depth_image)
        img_entropy = np.zeros(8,8)
        img_vs = np.vsplit(img,8)
        img_hs =[[]]*8
        #resize to 64*64
        for i in range(8):
            img_hs[i] = np.hsplit(img_vs[i],8)
        for i in range(8):
            for j in range(8):
                img_entropy[i][j] = self.depth_entropy(img_hs[i][j])
        max_entropy_value =  np.max(img_entropy)
        max_postion = np,where(img_entropy==max_entropy_value)
        return img_hs[max_postion[0][0]][max_postion[0][1]],max_postion[0],max_entropy_value


    def depth_entropy(self,depth_image):
        max_depth = np.max(img)
        min_depth = np.min(img)
        length = len(img)
        width =  len(img[0])
        temp = np.zeros(1,256)
        entropy = 0
        for i in range(length):
            for j in range(width):
                img[i][j]= int((img[i][j]-min_depth)*255/(max_depth-min_depth))
                val = img[i][j]
                temp[val] += 1
        for i in range(256):
            if temp[val] != 0:
                possible = float(temp[val])/(length*width)
                entropy + = -possible*math.log2(possible)
        return entropy
'''


        
                
                


                
            

