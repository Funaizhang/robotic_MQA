import math
import time
import h5py
import logging
import argparse
import numpy as np
import os, sys, json
from tqdm import tqdm

from scipy.misc import imread, imresize

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
from models import MultitaskCNN

import pdb

def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['questionIdxToToken'] = invert_dict(vocab['questionTokenToIdx'])
        vocab['answerIdxToToken'] = invert_dict(vocab['answerTokenToIdx'])

    assert vocab['questionTokenToIdx']['<NULL>'] == 0
    assert vocab['questionTokenToIdx']['<START>'] == 1
    assert vocab['questionTokenToIdx']['<END>'] == 2
    return vocab


def invert_dict(d):
    return {v: k for k, v in d.items()}


def _dataset_to_tensor(dset, mask=None, dtype=np.int64):
    arr = np.asarray(dset, dtype=dtype)
    if mask is not None:
        arr = arr[mask]
    if dtype == np.float32:
        tensor = torch.FloatTensor(arr)
    else:
        tensor = torch.LongTensor(arr)
    return tensor


def eqaCollateCnn(batch):
    transposed = list(zip(*batch))
    idx_batch = default_collate(transposed[0])
    question_batch = default_collate(transposed[1])
    answer_batch = default_collate(transposed[2])
    images_batch = default_collate(transposed[3])
    actions_in_batch = default_collate(transposed[4])
    actions_out_batch = default_collate(transposed[5])
    action_lengths_batch = default_collate(transposed[6])
    return [
        idx_batch, question_batch, answer_batch, images_batch,
        actions_in_batch, actions_out_batch, action_lengths_batch
    ]


def eqaCollateSeq2seq(batch):
    transposed = list(zip(*batch))
    idx_batch = default_collate(transposed[0])
    questions_batch = default_collate(transposed[1])
    answers_batch = default_collate(transposed[2])
    images_batch = default_collate(transposed[3])
    actions_in_batch = default_collate(transposed[4])
    actions_out_batch = default_collate(transposed[5])
    action_lengths_batch = default_collate(transposed[6])
    mask_batch = default_collate(transposed[7])

    return [
        idx_batch, questions_batch, answers_batch, images_batch,
        actions_in_batch, actions_out_batch, action_lengths_batch, mask_batch
    ]


class EqaDataset(Dataset):
    def __init__(self,
                 questions_h5,
                 vocab,
                 num_frames=1,
                 split='train',
                 gpu_id=0,
                 input_type='ques',
                 max_threads_per_gpu=10,
                 map_resolution=1000):

        self.questions_h5 = questions_h5
        self.vocab = load_vocab(vocab)
        np.random.seed()
        
        self.split = split
        self.gpu_id = gpu_id
        self.num_frames = num_frames

        self.input_type = input_type

        self.max_threads_per_gpu = max_threads_per_gpu
        self.map_resolution = map_resolution


        print('Reading question data into memory')
        self.questions = _dataset_to_tensor(questions_h5['questions'])
        self.answers = _dataset_to_tensor(questions_h5['answers'])
        self.actions = _dataset_to_tensor(questions_h5['actions'],dtype = np.float32)
        self.action_masks = _dataset_to_tensor(questions_h5['mask'])
        self.robot_positions = _dataset_to_tensor(questions_h5['robot_positions'],dtype = np.float32)
        self.action_images = questions_h5['images']
        self.action_lengths = _dataset_to_tensor(questions_h5['action_lengths'])

        #if input_type != 'ques':
        '''
        If training, randomly sample and load a subset of environments,
        train on those, and then cycle through to load the rest.

        On the validation and test set, load in order, and cycle through.

        For both, add optional caching so that if all environments
        have been cycled through once, then no need to re-load and
        instead, just the cache can be used.
        '''
        cnn_kwargs = {'num_classes': 191, 'pretrained': True}
        self.cnn = MultitaskCNN(**cnn_kwargs)
        self.cnn.eval()
        self.cnn.cuda()


    
    def __getitem__(self, index):
        # [VQA] question-only
        if self.input_type in ['pacman']:

            idx = index
            question = self.questions[index]
            #answer = self.answers[index]
            answer = self.answers[index]
            actions = self.actions[index]
            actions_masks = self.action_masks[index]
            robot_positions = self.robot_positions[index]
            action_lengths = self.action_lengths[index]


            if self.split in ['val', 'test']:    #return the data directly
                return (idx, question, answer, actions, robot_positions,action_lengths)  

            if self.split == 'train':                      #get iamge from data_set
                planner_images = self.action_images[index]
                planner_img_feats = self.cnn(
                        Variable(torch.FloatTensor(planner_images)
                                 .cuda())).data.cpu().numpy().copy()                              
                actions_in = actions.clone()
                actions_out = actions[1:].clone()
                actions_masks = actions_masks.clone()
                robot_positions = robot_positions.clone()
                     
            return (idx, question, answer, planner_img_feats,
                    actions_in, actions_out,
                   robot_positions, actions_masks,action_lengths)




    def __len__(self):
        if self.input_type == 'ques':
            return len(self.questions)
        else:
            return len(self.questions)


class EqaDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if 'questions_h5' not in kwargs:
            raise ValueError('Must give questions_h5')
        if 'vocab' not in kwargs:
            raise ValueError('Must give vocab')
        if 'input_type' not in kwargs:
            raise ValueError('Must give input_type')
        if 'split' not in kwargs:
            raise ValueError('Must give split')
        if 'gpu_id' not in kwargs:
            raise ValueError('Must give gpu_id')

        questions_h5_path = kwargs.pop('questions_h5')
        input_type = kwargs.pop('input_type')

        split = kwargs.pop('split')
        vocab = kwargs.pop('vocab')

        gpu_id = kwargs.pop('gpu_id')

        if 'max_threads_per_gpu' in kwargs:
            max_threads_per_gpu = kwargs.pop('max_threads_per_gpu')
        else:
            max_threads_per_gpu = 10

        if 'map_resolution' in kwargs:
            map_resolution = kwargs.pop('map_resolution')
        else:
            map_resolution = 1000

        if 'image' in input_type or 'cnn' in input_type:
            kwargs['collate_fn'] = eqaCollateCnn
        elif 'lstm' in input_type:
            kwargs['collate_fn'] = eqaCollateSeq2seq


        print('Reading questions from ', questions_h5_path)
        questions_h5 = h5py.File(questions_h5_path, 'r')
        self.dataset = EqaDataset(
            questions_h5,
            vocab,
            split=split,
            gpu_id=gpu_id,
            input_type=input_type,
            max_threads_per_gpu=max_threads_per_gpu,
            map_resolution=map_resolution,
            )

        super(EqaDataLoader, self).__init__(self.dataset, **kwargs)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_h5', default='data/04_22/train_v1.h5')
    parser.add_argument('-val_h5', default='data/04_22/val_v1.h5')
    parser.add_argument('-vocab_json', default='data/04_22/vocab_v1.json')

    parser.add_argument(
        '-input_type', default='ques', choices=['ques', 'ques,image'])
    parser.add_argument(
        '-num_frames', default=5,
        type=int)  # -1 = all frames of navigation sequence

    parser.add_argument('-batch_size', default=50, type=int)
    parser.add_argument('-max_threads_per_gpu', default=10, type=int)

    args = parser.parse_args()

    try:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
    except KeyError:
        print("CPU not supported")
        exit()

    train_loader_kwargs = {
        'questions_h5': args.train_h5,
        'vocab': args.vocab_json,
        'batch_size': args.batch_size,
        'input_type': args.input_type,
        'split': 'train',
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[0],
    }

    train_loader = EqaDataLoader(**train_loader_kwargs)



