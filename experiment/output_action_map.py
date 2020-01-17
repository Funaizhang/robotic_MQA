import time
import argparse
from datetime import datetime
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
from tqdm import tqdm
import h5py
import cv2 as cv
import time
import sys
import json
# sys.path.append(r'../train')
from models import actPlannerImproveModel, VqaLstmCnnAttentionModel,mapCNN
from data import load_vocab
from models import MultitaskCNN



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
                       )).data.cpu().numpy().copy()  
    planner_img_feats_var = Variable(torch.FloatTensor(planner_img_feats))

    planner_img_feats_var = planner_img_feats_var.unsqueeze(0)

    return position_in,planner_img_feats_var


def rgbd2tensor(rgb_image,depth_image):
        rgb_np = np.array(rgb_image)
        rgb_np = rgb_np.transpose(2,0,1)
        i = 0
        while i <3:
            rgb_mean = np.mean(rgb_np[i])
            rgb_std = np.std(rgb_np[i])
            if rgb_std == 0:         #error image
                index = index +1      #get the next image
                rgb_np = np.array(rgb_image)
                rgb_np = rgb_np.transpose(2,0,1)
                i=0
                continue
            rgb_miner = np.ones(rgb_np[i].shape)*rgb_mean
            rgb_np[i] = (rgb_np[i] - rgb_miner) / rgb_std
            i +=1
        

        rgb_tensor = Variable(torch.FloatTensor(rgb_np))

        # pre process depth_image
        dep_np = np.array(depth_image)
        dep_np = dep_np*65536/10000
        dep_np = np.clip(dep_np,0.0,1.2)   # the depth range: 0.0m -1.2m
        dep_mean = np.mean(dep_np)
        dep_std  = np.std(dep_np)

        dep_miner = np.ones(dep_np[i].shape)*dep_mean
        dep_np = (dep_np - dep_miner)/dep_std

        depth_tensor = Variable(torch.FloatTensor(dep_np)
        )        
        depth_tensor.unsqueeze(0)
        # print(depth_tensor.shape)
        depth_tensor = depth_tensor.repeat(3,1,1)


        depth_tensor = depth_tensor.unsqueeze(0)
        rgb_tensor = rgb_tensor.unsqueeze(0)

        return rgb_tensor,depth_tensor

def rgbd2heatmap(rgb_tensor,depth_tensor,cnn):
    map_possi = cnn(rgb_tensor,depth_tensor)
    map_possi = map_possi.view(2,20,32,32)
    map_possi_log = F.softmax(map_possi,dim=0)
    map_data = map_possi_log.data.numpy()   
    map_data = map_data[1]
    return map_data


def order2action(order):
    order_list =[
                [2.56,0],[25.6,0],[-2.56,0],[-25.6,0],
                [0,2.56],[0,25.6],[0,-2.56],[0,-25.6],[0,0]
                ]
    return order_list[order-1][0],order_list[order-1][1]
    
def tokenize(seq,delim=' ',punctToRemove=None,addStartToken=True,addEndToken=True):

    if punctToRemove is not None:
        for p in punctToRemove:
            seq = str(seq).replace(p, '')

    tokens = str(seq).split(delim)
    if addStartToken:
        tokens.insert(0, '<START>')

    if addEndToken:
        tokens.append('<END>')

    return tokens


def encode(seqTokens, tokenToIdx, allowUnk=False):
    seqIdx = []
    for token in seqTokens:
        if token not in tokenToIdx:
            if allowUnk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seqIdx.append(tokenToIdx[token])
    return seqIdx


def decode(seqIdx, idxToToken, delim=None, stopAtEnd=True):
    tokens = []
    for idx in seqIdx:
        tokens.append(idxToToken[idx])
        if stopAtEnd and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)
            

def test(rank):
    act_model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
    act_model = actPlannerImproveModel(**act_model_kwargs)
    act_checkpoint = torch.load(args.nav_weight, map_location='cpu')     #load checkpoint weights
    act_model.load_state_dict(act_checkpoint['state'])   #create model
    print('--- act_model loaded checkpoint ---')
    act_model.eval()


    res_model_dir = os.path.abspath("../train/models/resnet101.pth")
    my_map_cnn = mapCNN(checkpoint_path=res_model_dir)
    map_checkpoint = torch.load('mapcnn.pt', map_location='cpu')     #load checkpoint weights
    my_map_cnn.load_state_dict(map_checkpoint['state'])   #create map model
    my_map_cnn.eval()
    print('--- map_model loaded checkpoint ---')

    cnn_model_dir = os.path.abspath("../train/models/03_13_h3d_hybrid_cnn.pt")
    cnn_kwargs = {'num_classes': 191, 'pretrained': True,'checkpoint_path':cnn_model_dir}
    cnn = MultitaskCNN(**cnn_kwargs)
    cnn.eval()

    vocab_dir =  os.path.abspath("vocab.json")
    vocab_file = open(vocab_dir,'r',encoding='utf-8')
    vocab = json.load(vocab_file)


    planner_hidden = None
    max_action = 30
    position = [0,0]
    action_in_raw = [0]    #start action_in
    actions = []
    question = args.question
    print(question)
    questionTokens = tokenize(question, punctToRemove=['?'], addStartToken=False)
    
    encoded_question_raw = encode(questionTokens, vocab['questionTokenToIdx'])
    while(len(encoded_question_raw)<10):
        encoded_question_raw.append(0)                    #encode question
    encoded_question_raw = np.array(encoded_question_raw)
    encoded_question_tensor = _dataset_to_tensor(encoded_question_raw)
    encoded_question = Variable(encoded_question_tensor)
    encoded_question = encoded_question.unsqueeze(0)
    #print(encoded_question)
    action_times = 0
    push_signal = 0
    push_point = 0

    crop_w_offset = 470
    crop_w = 840
    crop_h_offset = 290
    crop_h = 690

    rgb_before = cv.imread(rgb_before_raw_dir)
    depth_before = cv.imread(depth_before_raw_dir)
    rgb_before_crop = rgb_before[crop_h_offset:crop_h_offset+crop_h, crop_w_offset:crop_w_offset+crop_w]
    depth_before_crop = depth_before[crop_h_offset:crop_h_offset+crop_h, crop_w_offset:crop_w_offset+crop_w]
    depth_before_crop = depth_before_crop[0]
    cv.imwrite(rgb_before_resize_dir,rgb_before_crop)
    
    rgb_dim = rgb_before.shape
    rgb_crop_dim = rgb_before_crop.shape
    # print(depth_dim)
    # print(depth_crop_dim)

    depth_before = depth_before[0]
    rgb_before_resize = cv.resize(rgb_before_crop,(256,256),interpolation=cv.INTER_AREA) 
    depth_before_resize = cv.resize(depth_before_crop,(256,256),interpolation=cv.INTER_AREA)

    rgb_tensor,depth_tensor = rgbd2tensor(rgb_before_resize,depth_before_resize)     #output_heatmap  
    heatmap_output = rgbd2heatmap(rgb_tensor,depth_tensor,my_map_cnn)
    f = h5py.File(heatmap_output_dir,'w')
    f['heatmap'] = heatmap_output

    heatmap_var = Variable(torch.FloatTensor(heatmap_output))
    heatmap_var = heatmap_var.unsqueeze(0)

    cv.imwrite(rgb_before_resize_dir,rgb_before_resize)
    # cv.imwrite(args.depth_image_before_dir_,depth_before_resize)

    while(action_times < max_action):

        #print(planner_img_feats_var.size())
        action_in_tensor = _dataset_to_tensor(action_in_raw)
        action_in = Variable(action_in_tensor)
        action_in = action_in.unsqueeze(0)
        action_in = action_in.unsqueeze(0)

        position_in,planner_img_feats_var = data2input(position,rgb_before,cnn)

        output_data, planner_hidden = act_model.planner_step(encoded_question,planner_img_feats_var,action_in,position_in,heatmap_var,planner_hidden)
        planner_possi = F.log_softmax(output_data, dim=1)
        planner_data = planner_possi.data.numpy()
        planner_data = planner_data[0]
        action_out = np.where(planner_data == np.max(planner_data))
        action_out = action_out[0][0]
        
        actions.append(action_out)          
        action_in_raw = [action_out]
        if action_out == 9:
            #print('stop')
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
        #action_position = position+position
        print('-- Suction at {} \n'.format(position))

        # convert to correct position
        # crop_position = (int(position[0] / 256 * rgb_dim[1] - crop_w_offset),
        #                 int(position[1] / 256 * rgb_dim[0] - crop_h_offset))
        crop_position = (int(position[0] / 256 * rgb_crop_dim[1]),
                        int(position[1] / 256 * rgb_crop_dim[0]))

        # draw a red cross at position on cropped rgb
        rgb_before_res = cv.drawMarker(rgb_before_crop,
                                        crop_position, 
                                        (0,0,255),
                                        markerType=cv.MARKER_CROSS,
                                        markerSize=50,
                                        thickness=5,
                                        line_type=cv.LINE_AA)
        cv.imwrite(rgb_before_res_dir,rgb_before_res)

        # draw the same on rgb_before_resize for comparison
        # convert to correct position
        crop_position_ = (int(position[0]), int(position[1]))
        # draw a red cross at position on cropped rgb
        rgb_before_resize = cv.drawMarker(rgb_before_resize,
                                        crop_position_, 
                                        (0,0,255),
                                        markerType=cv.MARKER_CROSS,
                                        markerSize=10,
                                        thickness=5,
                                        line_type=cv.LINE_AA)
        cv.imwrite(rgb_before_resize_dir,rgb_before_resize)

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
        #action_position = position_start+position_end
        print('-- Push from {} to {} \n'.format(position_start,position_end))

        # convert to correct position
        # crop_position_start = (int(position_start[0] / 256 * rgb_dim[1] - crop_w_offset), 
        #                         int(position_start[1] / 256 * rgb_dim[0] - crop_h_offset))
        # crop_position_end = (int(position_end[0] / 256 * rgb_dim[1] - crop_w_offset), 
        #                         int(position_end[1] / 256 * rgb_dim[0] - crop_h_offset))
        crop_position_start = (int(position_start[0] / 256 * rgb_crop_dim[1]), 
                                int(position_start[1] / 256 * rgb_crop_dim[0]))
        crop_position_end = (int(position_end[0] / 256 * rgb_crop_dim[1]), 
                                int(position_end[1] / 256 * rgb_crop_dim[0]))

        # draw a red, 10pt arrow from position_start to position_end on cropped rgb
        rgb_before_res = cv.arrowedLine(rgb_before_crop, 
                                        crop_position_start, 
                                        crop_position_end, 
                                        (0,0,255),
                                        thickness=3,
                                        line_type=cv.LINE_AA)
        cv.imwrite(rgb_before_res_dir,rgb_before_res)


        # draw the same on rgb_before_resize for comparison
        # convert to correct position
        crop_position_start_ = (int(position_start[0]), int(position_start[1]))
        crop_position_end_ = (int(position_end[0]), int(position_end[1]))

        # draw a red cross at position on cropped rgb
        rgb_before_resize = cv.arrowedLine(rgb_before_resize, 
                                        crop_position_start_, 
                                        crop_position_end_, 
                                        (0,0,255),
                                        thickness=3,
                                        line_type=cv.LINE_AA)
        cv.imwrite(rgb_before_resize_dir,rgb_before_resize)

    else:
        print('-- No action \n')
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('-nav_weight', default='act2_map.pt')
    parser.add_argument('-vqa_weight', default='vqa.pt')
    parser.add_argument('-vocab_json', default= "vocab.json")
    parser.add_argument('-question',default= "what is under the box?")
    parser.add_argument('-scene_index', default= "0000")
    # parser.add_argument('-rgb_image_before_dir',default= 'rgb_before/rgb_before.jpg')
    # parser.add_argument('-rgb_image_after_dir',default= 'rgb_after/rgb_after.jpg')
    # parser.add_argument('-depth_image_before_dir',default= 'depth_before/depth_before.png')
    # parser.add_argument('-depth_image_after_dir',default= 'depth_after/depth_after.png')
    # parser.add_argument('-rgb_image_before_dir_',default= 'rgb_before_/rgb_before_.jpg')
    # parser.add_argument('-rgb_image_after_dir_',default= 'rgb_after_/rgb_after_.jpg')
    # parser.add_argument('-rgb_crop_before_dir_',default= 'rgb_before_/rgb_before_crop.jpg')
    # parser.add_argument('-rgb_crop_after_dir_',default= 'rgb_after_/rgb_after_crop.jpg')
    # parser.add_argument('-depth_image_before_dir_',default= 'depth_before_/depth_before_.png')
    # parser.add_argument('-depth_image_after_dir_',default= 'depth_after_/depth_after_.png')
    # parser.add_argument('-heatmap_output_dir',default= 'heatmap/heatmap_before.h5')
    args = parser.parse_args()

    rgb_before_raw_dir = 'rgb_before/{}_color.jpg'.format(args.scene_index)
    rgb_after_raw_dir = 'rgb_after/{}_color.jpg'.format(args.scene_index)
    depth_before_raw_dir = 'depth_before/{}_depth_colored.png'.format(args.scene_index)
    depth_after_raw_dir = 'depth_after/{}_depth_colored.png'.format(args.scene_index)
    rgb_before_resize_dir = 'rgb_before_/{}_color_crop.jpg'.format(args.scene_index)
    rgb_before_res_dir = 'rgb_before_/{}_color_res.jpg'.format(args.scene_index)
    heatmap_output_dir = 'heatmap/{}_heatmap_before.h5'.format(args.scene_index)

    test(0)



