'''
generate best path for the imitation learning
'''

import json
import os
import pickle
import sys
import numpy as np
import math



class shortest_path():
#init with the current scene    
    def __init__(self,env):
        self.my_env = env
        self.shortest_path_dir = os.path.abspath('path/path.pk')
        self.question_dir = os.path.abspath('question/question.json')
        question_file = open(self.question_dir,encoding='utf-8')
        self.question_dic = json.load(self.question_file)
        question_file.close()
        self.obj_position,self.obj_oritentation  = self.env.ur5.get_obj_positions_and_orientations
        self.obj_name = self.env.ur5.obj_name
        self.most_action  =5

    def generate_shortest_path(self):
        obj_order = 0
        all_path = []
        for ques in self.question_dic:
            episode_rgb_images=[]
            episode_depth_image=[]
            action_type= []
            actions = []
            ans = ques['answer']
            question = ques['question']

            if ques['type'] is 'exist_positive':
                action_position = []
                action_times = 0
                action_position,act_type = self.is_targetobject_covered(obj_order)
                while (len(action_position)>0) and (action_times<self.most_action):     #such until no covering
                    action_type.append(act_type)
                    actions.append(action_position)
                    self.env.action(action_position)                
                    rgb_image,depth_image = self.env.camera.get_camera_data()
                    eposide_rgb_images.append(rgb_image)
                    eposide_depth_images.append(depth_image)
                    action_position,act_type = self.is_targetobject_covered(obj_order)
                    action_times += 1

            else if ques['type'] is 'exist_negative':
                action_position = []
                action_times =0
                action_position ,act_type= self.findclutter()
                while (len(action_position)>0) and (action_times<self.most_action):     #push until no clutter
                    action_type.append(act_type)
                    actions.append(action_position)
                    self.env.action(action_position)                
                    rgb_image,depth_image = self.env.camera.get_camera_data()
                    eposide_rgb_images.append(rgb_image)
                    eposide_depth_images.append(depth_image)
                    action_position,act_type = self.findclutter()
                    action_times += 1

            result = {
                        "actions": actions,
                        "act_type": action_type,
                        "answer": ans
                        "rgb_images": episode_rgb_images,
                        "depth_images": episode_depth_images,
                        "question": question,
                    }
            all_path.append(result)
        with open(self.shortest_path_dir, "wb") as f:
            pickle.dump(result, f)
    
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



        
                
                


                
            

