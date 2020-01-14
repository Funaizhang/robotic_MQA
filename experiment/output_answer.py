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
from models import actPlannerBaseModel, VqaLstmCnnAttentionModel,mapCNN
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

def _dataset_to_tensor(dset, mask=None, dtype=np.int64):
    arr = np.asarray(dset, dtype=dtype)
    if mask is not None:
        arr = arr[mask]
    if dtype == np.float32:
        tensor = torch.FloatTensor(arr)
    else:
        tensor = torch.LongTensor(arr)
    return tensor

def data2input(rgb_image_raw):
     
    shrink = cv.resize(rgb_image_raw,(224,224),interpolation=cv.INTER_AREA)
    shrink = np.array(shrink)
    shrink = shrink.transpose((2,0,1))
    shrink = shrink.reshape(1,3,224,224)
    shrink = (shrink/255.0).astype(np.float32)
                  #resize image 
    '''
    planner_img_feats = cnn(
                        Variable(torch.FloatTensor(shrink)
                       )).data.cpu().numpy().copy()
    '''  
   

    return shrink

def rgbd2heatmap(rgb_tensor,depth_tensor,cnn):
    map_possi = cnn(rgb_tensor,depth_tensor)
    map_possi = map_possi.view(2,20,32,32)
    map_possi_log = F.softmax(map_possi,dim=0)
    map_data = map_possi_log.data.numpy()   
    map_data = map_data[1]
    return map_data

    
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



    cnn_model_dir = os.path.abspath("../train/models/03_13_h3d_hybrid_cnn.pt")

    vqa_model_kwargs = {'vocab': load_vocab(args.vocab_json),'checkpoint_path':cnn_model_dir}
    vqa_model = VqaLstmCnnAttentionModel(**vqa_model_kwargs)
    vqa_checkpoint = torch.load(args.vqa_weight)     #load checkpoint weights
    vqa_model.load_state_dict(vqa_checkpoint['state'])
    print('--- vqa_model loaded checkpoint ---')


    res_model_dir = os.path.abspath("../train/models/resnet101.pth")
    my_map_cnn = mapCNN(checkpoint_path=res_model_dir)
    map_checkpoint = torch.load('mapcnn.pt', map_location='cpu')     #load checkpoint weights
    my_map_cnn.load_state_dict(map_checkpoint['state'])   #create map model
    print('--- map_model loaded checkpoint ---')

    cnn_kwargs = {'num_classes': 191, 'pretrained': True,'checkpoint_path':cnn_model_dir}
    cnn = MultitaskCNN(**cnn_kwargs)
    cnn.eval()

    vocab_dir =  os.path.abspath("vocab.json")
    vocab_file = open(vocab_dir,'r',encoding='utf-8')
    vocab = json.load(vocab_file)

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

    rgb_before = cv.imread(args.rgb_image_before_dir)
    rgb_after = cv.imread(args.rgb_image_after_dir)
    depth_after = cv.imread(args.depth_image_after_dir)
    depth_after = depth_after[0]
    depth_dim = depth_after.shape
    print(depth_dim)

    rgb_after_resize = cv.resize(rgb_after,(256,256),interpolation=cv.INTER_AREA)
    # crop and add marking
    depth_after_resize = cv.resize(depth_after,(256,256),interpolation=cv.INTER_AREA)
    # crop and add marking

    rgb_tensor,depth_tensor = rgbd2tensor(rgb_after_resize,depth_after_resize)     #output_heatmap  
    heatmap_output = rgbd2heatmap(rgb_tensor,depth_tensor,my_map_cnn)
    f = h5py.File(args.heatmap_output_dir,'w')
    f['heatmap'] = heatmap_output

    cv.imwrite(args.rgb_image_after_dir,rgb_after_resize)
    cv.imwrite(args.depth_image_after_dir,depth_after_resize)


    before_image_feat = data2input(rgb_before)
    after_image_feat =  data2input(rgb_after_resize)

    input_image = [before_image_feat,after_image_feat]
    input_image_feats = Variable(torch.FloatTensor(input_image))
    input_image_feats = input_image_feats.view(1,2,3,224,224)

    # print(input_image_feats.size())

            

    #print(input_image.size())
    #print(before_image_feat.size())
  

    scores, _ = vqa_model(input_image_feats, encoded_question)
    scores = scores.data.numpy()
    scores = scores[0]
    answer_predict = np.where(scores == np.max(scores))
    answer_predict = answer_predict[0][0]
    answer_dic = vocab["answerTokenToIdx"]
    answer =  [k for k,v in answer_dic.items() if v == answer_predict ]
    

    print(answer[0])



            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('-nav_weight', default='act.pt')
    parser.add_argument('-vqa_weight', default='vqa.pt')
    parser.add_argument('-vocab_json', default= "vocab.json")
    parser.add_argument('-question',default= "is there a key on the table?")
    parser.add_argument('-rgb_image_before_dir',default= 'rgb_before/rgb_before.jpg')
    parser.add_argument('-rgb_image_after_dir',default= 'rgb_after/rgb_after.jpg')
    parser.add_argument('-depth_image_before_dir',default= 'depth_before/depth_before.png')
    parser.add_argument('-depth_image_after_dir',default= 'depth_after/depth_after.png')
    parser.add_argument('-heatmap_output_dir',default= 'heatmap/heatmap_after.h5')
    args = parser.parse_args()
    test(0)



