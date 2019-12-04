import time
import argparse
from datetime import datetime
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from models import NavPlannerControllerModel, VqaLstmCnnAttentionModel
from data import EqaDataLoader
from metrics import NavMetric
from data import load_vocab
from torch.autograd import Variable
from tqdm import tqdm
from models import MultitaskCNN
import cv2 as cv
import time
import sys
import time
sys.path.append(r'../simulation')
import enviroment
from generate_questions import Qusetion


torch.backends.cudnn.enabled = False

################################################################################################
#make models trained in pytorch 4 compatible with earlier pytorch versions
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

################################################################################################



def _dataset_to_tensor(dset, mask=None, dtype=np.int64):
    arr = np.asarray(dset, dtype=dtype)
    if mask is not None:
        arr = arr[mask]
    if dtype == np.float32:
        tensor = torch.FloatTensor(arr)
    else:
        tensor = torch.LongTensor(arr)
    return tensor

def data2input(position,rgb_image_raw,cnn):
    position_in_tensor = _dataset_to_tensor(position,dtype=np.float32)   #positions
    position_in = Variable(position_in_tensor)
    position_in = position_in.unsqueeze(0)
    position_in = position_in.unsqueeze(0)
      
    shrink = cv.resize(rgb_image_raw,(224,224),interpolation=cv.INTER_AREA)
    shrink = np.array(shrink)
    shrink = shrink.transpose((2,0,1))
    shrink = shrink.reshape(1,3,224,224)
    shrink = (shrink/255.0).astype(np.float32)
                  #resize image 
    planner_img_feats = cnn(
                        Variable(torch.FloatTensor(shrink)
                        .cuda())).data.cpu().numpy().copy()  
    planner_img_feats_var = Variable(torch.FloatTensor(planner_img_feats))

    planner_img_feats_var = planner_img_feats_var.unsqueeze(0)

    return position_in,planner_img_feats_var


def order2action(order):
    order_list =[
                [2.56,0],[25.6,0],[-2.56,0],[-25.6,0],
                [0,2.56],[0,25.6],[0,-2.56],[0,-25.6],[0,0]
                ]
    return order_list[order-1][0],order_list[order-1][1]
    
            

def test(rank):
    nav_model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
    nav_model = NavPlannerControllerModel(**nav_model_kwargs)
    nav_checkpoint = torch.load(args.nav_weight)     #load checkpoint weights
    nav_model.load_state_dict(nav_checkpoint['state'])   #create model
    print('--- nav_model loaded checkpoint ---')

    cnn_kwargs = {'num_classes': 191, 'pretrained': True}
    cnn = MultitaskCNN(**cnn_kwargs)
    cnn.eval()
    cnn.cuda()                   #create cnn model

    vqa_model_kwargs = {'vocab': load_vocab(args.vocab_json)}
    vqa_model = VqaLstmCnnAttentionModel(**vqa_model_kwargs)
    vqa_checkpoint = torch.load(args.vqa_weight)     #load checkpoint weights
    vqa_model.load_state_dict(vqa_checkpoint['state'])
    print('--- vqa_model loaded checkpoint ---')

    # need cnn?

    scene = "test-10-obj-100.txt"
    my_env = enviroment.Environment(is_testing=0,testing_file=scene)
    object_exist_list = my_env.ur5.object_type
    print("Objetcts that exist: ")
    print(object_exist_list)                    #create simulation enviroment

    my_question = Qusetion(object_exist_list)   #create testing question
    testing_questions = my_question.createQueue()
    vocab = my_question.create_vocab()


    for question in testing_questions:       
        planner_hidden = None
        max_action = 30
        position = [0,0]
        action_in_raw = [0]    #start action_in
        actions = []
 
        print(question['question'])   #question 
        questionTokens = my_question.tokenize(question['question'], punctToRemove=['?'], addStartToken=False)
        encoded_question_raw = my_question.encode(questionTokens, vocab['questionTokenToIdx'])
        encoded_question_raw.append(0)                     #encode question
        encoded_question_raw = np.array(encoded_question_raw)
        encoded_question_tensor = _dataset_to_tensor(encoded_question_raw)
        encoded_question = Variable(encoded_question_tensor)
        encoded_question = encoded_question.unsqueeze(0)
        print(encoded_question)
        action_times = 0
        push_signal = 0
        push_point = 0
        
        while(action_times < max_action):

            #print(planner_img_feats_var.size())
            action_in_tensor = _dataset_to_tensor(action_in_raw)
            action_in = Variable(action_in_tensor)
            action_in = action_in.unsqueeze(0)
            action_in = action_in.unsqueeze(0)

            _,rgb_image_raw = my_env.camera.get_camera_data()   #before
            position_in,planner_img_feats_var = data2input(position,rgb_image_raw,cnn)

            output_data, planner_hidden = nav_model.planner_step(encoded_question,planner_img_feats_var,action_in,position_in,planner_hidden)
            planner_possi = F.log_softmax(output_data, dim=1)
            planner_data = planner_possi.data.numpy()
            planner_data = planner_data[0]
            action_out = np.where(planner_data == np.max(planner_data))
            action_out = action_out[0][0]
           
            actions.append(action_out)          
            action_in_raw = [action_out]
            if action_out == 9:
                print('stop')
                break
            elif action_out ==0:
                push_signal = 1
                push_point = action_times
            else:
                dx,dy = order2action(action_out)
                position[0] += dx
                position[1] += dy
            action_times +=1
        
        if len(actions)>2 and push_signal == 0:
            action_position = position+position
            my_env.UR5_action(action_position,2)  #sucking
        elif len(actions)>2 and push_signal ==1:   #pushing
            position_start=[0,0]
            position_end =[0,0]
            for i in range(len(actions)):
                if i<=push_point:   #the first step
                    dx,dy = order2action(actions[i])
                    position_start[0] += dx
                    position_start[1] += dy
                    position_end[0] +=dx
                    position_end[1] += dy
                else:  #the second step
                    dx,dy = order2action(actions[i])
                    position_end[0] += dx
                    position_end[1] += dy
            action_position = position_start+position_end
            my_env.UR5_action(action_position,1)  #pushing

        # get image after actions
        _,rgb_image_after = my_env.camera.get_camera_data()     # image after actions
        shrink = cv.resize(rgb_image_raw,(224,224),interpolation=cv.INTER_AREA)
        shrink = np.array(shrink)
        shrink = shrink.transpose((2,0,1))
        shrink = shrink.reshape(1,3,224,224)
        shrink= (shrink/255.0).astype(np.float32)
        images = torch.FloatTensor(shrink)
        images = Variable(images)
        images = images.unsqueeze(0)

        # process images

        # answer question in vqa now
        # encoded_question already done

        scores, _ = vqa_model(images, encoded_question)
        scores = scores.data.numpy()
        scores = scores[0]
        answer_predict = np.where(scores == np.max(scores))
        answer_predict = answer_predict[0][0]
        if answer_predict == 0:
            print('--- Predict: Exists not')
        elif answer_predict == 1:
            print('--- Predict: Exists')
        else:
            raise Exception('Prediction neither 0 nor 1')
        # accuracy TODO, not needed now


            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('-nav_weight', default='nav.pt')
    parser.add_argument('-vqa_weight', default='vqa.pt')
    parser.add_argument('-vocab_json', default='vocab.json')
    args = parser.parse_args()
    test(0)



