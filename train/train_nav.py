import time
import argparse
from datetime import datetime
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from models import NavPlannerControllerModel
from data import EqaDataLoader
from metrics import NavMetric
from models import MaskedNLLCriterion
from models import get_state, ensure_shared_grads
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
    planner_img_feats_var =Variable(torch.FloatTensor(planner_img_feats))

    planner_img_feats_var =  planner_img_feats_var.unsqueeze(0)

    return position_in,planner_img_feats_var


def order2action(order):
    order_list =[
                [2.56,0],[25.6,0],[-2.56,0],[-25.6,0],
                [0,2.56],[0,25.6],[0,-2.56],[0,-25.6],[0,0]
                ]
    return order_list[order-1][0],order_list[order-1][1]
    
            

def test(rank, test_model_dir):
     model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
     model = NavPlannerControllerModel(**model_kwargs)
     checkpoint  = torch.load(test_model_dir)     #load check point
     model.load_state_dict(checkpoint['state'])   #create model

     cnn_kwargs = {'num_classes': 191, 'pretrained': True}
     cnn = MultitaskCNN(**cnn_kwargs)
     cnn.eval()
     cnn.cuda()                   #create cnn model


     scene = "test-10-obj-00.txt"
     my_env = enviroment.Environment(is_testing=1,testing_file = scene)
     object_exist_list = my_env.ur5.object_type
     print("the objetct which is exist:")
     print(object_exist_list)                    #create simulation enviroment

     my_question =Qusetion(object_exist_list)   #create testing question
     testing_questions  = my_question.createQueue()
     vocab = my_question.create_vocab()


     for question in testing_questions:       
        planner_hidden = None
        max_action = 30
        position = [0,0]
        action_in_raw =[0]    #start action_in
        actions = []
 
        print(question['question'])   #question 
        questionTokens = my_question.tokenize(
                question['question'], punctToRemove=['?'], addStartToken=False)
        encoded_question_raw = my_question.encode(questionTokens, vocab['questionTokenToIdx'])
        encoded_question_raw.append(0)                     #encode question
        encoded_question_raw = np.array(encoded_question_raw)
        encoded_question_tensor = _dataset_to_tensor(encoded_question_raw)
        encoded_question = Variable(encoded_question_tensor)
        encoded_question = encoded_question.unsqueeze(0)
        action_times = 0
        
        while(action_times < max_action):

            #print(planner_img_feats_var.size())
            action_in_tensor = _dataset_to_tensor(action_in_raw)
            action_in = Variable(action_in_tensor)
            action_in = action_in.unsqueeze(0)
            action_in = action_in.unsqueeze(0)

            _,rgb_image_raw = my_env.camera.get_camera_data()  
            position_in,planner_img_feats_var =data2input(position,rgb_image_raw,cnn)

            output_data, planner_hidden = model.planner_step(encoded_question,planner_img_feats_var,action_in,position_in,planner_hidden)
            planner_possi = F.log_softmax(output_data, dim=1)
            planner_data =  planner_possi.data.numpy()
            planner_data = planner_data[0]
            action_out = np.where(planner_data == np.max(planner_data))
            action_out = action_out[0][0]
           
            actions.append(action_out)          
            action_in_raw = [action_out]
            if action_out == 9:
                print('stop')
                break
            else:
                dx,dy = order2action(action_out)
                position[0] += dx
                position[1] += dy
            action_times +=1
        
        if len(actions)>2 and len(actions)<20:
            action_position = position+position
            my_env.UR5_action(action_position,2)  #sucking
        elif len(actions)>=20:   #pushing
            position_start=[0,0]
            position_end =[0,0]
            for i in range(len(actions)):
                if i<len(actions)/2:   #the first step
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

            
            
           

        

 
        








def train(rank, args, shared_model):
    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))
    if args.model_type == 'pacman':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        model = NavPlannerControllerModel(**model_kwargs)

    else:
        exit()

   

    optim = torch.optim.Adamax(
        filter(lambda p: p.requires_grad, shared_model.parameters()),
        lr=args.learning_rate)

    train_loader_kwargs = {
        'questions_h5': args.train_h5,
        'vocab': args.vocab_json,
        'batch_size': args.batch_size,
        'input_type': args.model_type,
        'split': 'train',
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[rank % len(args.gpus)]
    }

    args.output_log_path = os.path.join(args.log_dir,
                                        'train_' + str(rank) + '.json')

    if 'pacman' in args.model_type:

        metrics = NavMetric(
            info={'split': 'train',
                  'thread': rank},
            metric_names=['planner_loss', 'controller_loss'],
            log_json=args.output_log_path)

    else:

        metrics = NavMetric(
            info={'split': 'train',
                  'thread': rank},
            metric_names=['loss'],
            log_json=args.output_log_path)

    train_loader = EqaDataLoader(**train_loader_kwargs)

    print('train_loader has %d samples' % len(train_loader.dataset))
    logging.info('TRAIN: train loader has {} samples'.format(len(train_loader.dataset)))

    t, epoch = 0, 0

    while epoch < int(args.max_epochs):
        planner_lossFn = MaskedNLLCriterion().cuda()
        for batch in train_loader:
            t += 1
            model.load_state_dict(shared_model.state_dict())
            model.train()
            model.cuda()

            idx, questions,_,planner_img_feats, planner_actions_in, planner_actions_out, planner_positions, planner_masks,planner_action_lengths  = batch

            # calcualte var of input data(qustion,action,image)  
            questions_var = Variable(questions.cuda())
            planner_img_feats_var = Variable(planner_img_feats.cuda())
            planner_actions_in_var = Variable(
                planner_actions_in.cuda())
            planner_actions_out_var = Variable(
                planner_actions_out.cuda())
            planner_positions_var = Variable(
            planner_positions.cuda())
            planner_masks_var = Variable(planner_masks.cuda())
            planner_action_lengths = planner_action_lengths.cuda()    #


            # find the question and image that need most action
            
            planner_action_lengths, perm_idx = planner_action_lengths.sort(
                0, descending=True) 

            questions_var = questions_var[perm_idx]

            planner_img_feats_var = planner_img_feats_var[perm_idx]
            planner_actions_in_var = planner_actions_in_var[perm_idx]
            planner_actions_out_var = planner_actions_out_var[perm_idx]
            planner_masks_var = planner_masks_var[perm_idx]
            planner_positions_var = planner_positions_var[perm_idx]

            '''
            print('action')
            print(planner_actions_out_var)
            print('position')
            print(planner_positions_var)
            print('image')
            print(planner_img_feats_var)
            '''

            #print(planner_masks_var)

            planner_scores, planner_hidden = model(      
                questions_var, planner_img_feats_var,
                planner_actions_in_var,
                planner_positions_var, planner_action_lengths.cpu().numpy().astype(np.long))

            planner_logprob = F.log_softmax(planner_scores, dim=1)


            
            planner_loss = planner_lossFn(
                        planner_logprob,
                        planner_actions_out_var.contiguous().view(-1, 1),
                        planner_masks_var.contiguous().view(-1, 1))
            

            '''
            planner_loss = planner_lossFn(
                planner_logprob.view(-1,21,2),
                planner_actions_out_var.float())
            planner_loss = planner_loss.mean(2)

            print('loss')
            print(planner_loss)
            print('masks')
            print(planner_masks_var.float()[:,:-1])
            '''

            #planner_loss = planner_loss * planner_masks_var.float()[:,:-1]
            #planner_loss = planner_loss.mean()
            #TODO masked


            # zero grad
            optim.zero_grad()

            # update metrics
            print("TRAINING PACMAN planner-loss:{}".format(planner_loss.item()))
            logging.info("TRAINING PACMAN planner-loss:{}".format(planner_loss.item()))

            # backprop and update
            (planner_loss).backward()


            ensure_shared_grads(model.cpu(), shared_model)
            optim.step()

            #if t % args.print_every == 0:
            #    print(metrics.get_stat_string())
            #    logging.info("TRAIN: metrics: {}".format(metrics.get_stat_string()))

        epoch += 1

        if epoch % args.save_every == 0:

            model_state = get_state(model)
            optimizer_state = optim.state_dict()

            aad = dict(args.__dict__)
            ad = {}
            for i in aad:
                if i[0] != '_':
                    ad[i] = aad[i]

            checkpoint = {'args': ad,
                    'state': model_state,
                    'epoch': epoch,
                    'optimizer': optimizer_state}

            checkpoint_path = '%s/epoch_%d_thread_%d.pt' % (
                args.checkpoint_dir, epoch, rank)
            print('Saving checkpoint to %s' % checkpoint_path)
            logging.info("TRAIN: Saving checkpoint to {}".format(checkpoint_path))
            torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('-train_h5', default='scene_all.h5')
    parser.add_argument('-val_h5', default='data/val.h5')
    parser.add_argument('-test_h5', default='data/test.h5')

    parser.add_argument('-vocab_json', default='vocab.json')


    parser.add_argument(
        '-mode',
        default='train',
        type=str,
        choices=['train', 'eval', 'train+eval','test'])
    parser.add_argument('-eval_split', default='val', type=str)

    # model details
    parser.add_argument(
        '-model_type',
        default='pacman',
        choices=['cnn', 'cnn+q', 'lstm', 'lstm+q', 'lstm-mult+q', 'pacman'])
    parser.add_argument('-max_episode_length', default=100, type=int)
    parser.add_argument('-curriculum', default=0, type=int)

    # optim params
    parser.add_argument('-batch_size', default=128, type=int)
    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-max_epochs', default=10000, type=int)


    # bookkeeping
    parser.add_argument('-print_every', default=5, type=int)
    parser.add_argument('-eval_every', default=1, type=int)
    parser.add_argument('-save_every', default=100, type=int) #optional if you would like to save specific epochs as opposed to relying on the eval thread
    parser.add_argument('-identifier', default='cnn')
    parser.add_argument('-num_processes', default=1, type=int)
    parser.add_argument('-max_threads_per_gpu', default=10, type=int)

    # checkpointing
    parser.add_argument('-checkpoint_path', default=False)
    parser.add_argument('-checkpoint_dir', default='checkpoints/nav/')
    parser.add_argument('-log_dir', default='logs/nav/')
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-cache', default=False, action='store_true')
    parser.add_argument('-max_controller_actions', type=int, default=5)
    parser.add_argument('-max_actions', type=int)
    args = parser.parse_args()
    
    args.train_h5 = os.path.abspath(args.train_h5)
    args.time_id = time.strftime("%m_%d_%H:%M")

    #MAX_CONTROLLER_ACTIONS = args.max_controller_actions

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    if args.curriculum:
        assert 'lstm' in args.model_type #TODO: Finish implementing curriculum for other model types

    logging.basicConfig(filename=os.path.join(args.log_dir, "run_{}.log".format(
                                                str(datetime.now()).replace(' ', '_'))),
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    try:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
    except KeyError:
        print("CPU not supported")
        logging.info("CPU not supported")
        exit()

    if args.checkpoint_path != False:

        print('Loading checkpoint from %s' % args.checkpoint_path)
        logging.info("Loading checkpoint from {}".format(args.checkpoint_path))

        args_to_keep = ['model_type']

        checkpoint = torch.load(args.checkpoint_path, map_location={
            'cuda:0': 'cpu'
        })

        for i in args.__dict__:
            if i not in args_to_keep:
                checkpoint['args'][i] = args.__dict__[i]

        args = type('new_dict', (object, ), checkpoint['args'])

    args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                       args.time_id + '_' + args.identifier)
    args.log_dir = os.path.join(args.log_dir,
                                args.time_id + '_' + args.identifier)

    print(args.__dict__)
    logging.info(args.__dict__)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        os.makedirs(args.log_dir)


    if args.model_type == 'pacman':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        shared_model = NavPlannerControllerModel(**model_kwargs)

    else:

        exit()

    shared_model.share_memory()

    if args.checkpoint_path != False:
        print('Loading params from checkpoint: %s' % args.checkpoint_path)
        logging.info("Loading params from checkpoint: {}".format(args.checkpoint_path))
        shared_model.load_state_dict(checkpoint['state'])

    if args.mode == 'eval':

        eval(0, args, shared_model)

    elif args.mode == 'train':

        if args.num_processes > 1:
            processes = []
            for rank in range(0, args.num_processes):
                # for rank in range(0, args.num_processes):
                p = mp.Process(target=train, args=(rank, args, shared_model))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        else:
            train(0, args, shared_model)

    elif args.mode == 'test':
        test_model_dir = "epoch_4900_thread_0.pt"
        test(0,test_model_dir)

    else:
        processes = []

        # Start the eval thread
        p = mp.Process(target=eval, args=(0, args, shared_model))
        p.start()
        processes.append(p)

        # Start the training thread(s)
        for rank in range(1, args.num_processes + 1):
            # for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
