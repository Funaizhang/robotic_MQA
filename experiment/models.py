# Model defs for navigation and question answering
# Navigation: CNN, LSTM, Planner-controller
# VQA: question-only, 5-frame + attention

import time
import h5py
import math
import argparse
import numpy as np
import os, sys, json

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pdb

def build_mlp(input_dim,
              hidden_dims,
              output_dim,
              use_batchnorm=False,
              dropout=0,
              add_sigmoid=1):
    layers = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    for dim in hidden_dims:
        layers.append(nn.Linear(D, dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU(inplace=True))
        D = dim
    layers.append(nn.Linear(D, output_dim))

    if add_sigmoid == 1:
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)





def get_state(m):
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state


def repackage_hidden(h, batch_size):
    # wraps hidden states in new Variables, to detach them from their history
    if type(h) == Variable:
        return Variable(
            h.data.resize_(h.size(0), batch_size, h.size(2)).zero_())
    else:
        return tuple(repackage_hidden(v, batch_size) for v in h)


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class MaskedNLLCriterion(nn.Module):
    def __init__(self):
        super(MaskedNLLCriterion, self).__init__()

    def forward(self, input, target, mask):

        logprob_select = torch.gather(input, 1, target)

        out = torch.masked_select(logprob_select, mask)

        loss = -torch.sum(out) / mask.float().sum()
        return loss

class mapCNN(nn.Module):
    def __init__(self,
               num_classes=40,     #2*20
               checkpoint_path='models/resnet101.pth'
    ):
        super(mapCNN, self).__init__()

        self.num_classes = num_classes
        res_model = torchvision.models.resnet101(pretrained=False)
        #print(res_model)
        res_model.load_state_dict(torch.load(checkpoint_path))
        
        self.rgb_layer = torch.nn.Sequential(*list(res_model.children())[:-4])    #remove fc layer
        
        
        res_model1 =  torchvision.models.resnet101(pretrained=False)
        res_model1.load_state_dict(torch.load(checkpoint_path))

        self.depth_layer = torch.nn.Sequential(*list(res_model1.children())[:-4])    #remove fc layer
        

        print('Loading resnet weights from %s' % checkpoint_path)


        self.conv_layer = nn.Sequential(
            nn.Conv2d(1024,512,1),
            nn.Conv2d(512,128,1),
            nn.Conv2d(128,self.num_classes,1)
        )

        for param in self.rgb_layer[1].parameters():  #fix bn
            param.requires_grad = False

        for param in self.depth_layer[1].parameters():
            param.requires_grad = False

        for name,param in self.named_parameters():   
            #if name in ['bn1','bn2','bn3']:
            if 'bn1' in name or 'bn2' in name or 'bn3' in name:
                param.requires_grad = False




    def forward(self,rgb_image,depth_image):
        output1 = self.rgb_layer(rgb_image)
        output2 = self.depth_layer(depth_image)
        input_conv = torch.cat([output1, output2], 1)
        output = self.conv_layer(input_conv)

        return output
        
        

        


class MultitaskCNNOutput(nn.Module):
    def __init__(
            self,
            num_classes=191,
            pretrained=True,
            checkpoint_path='models/03_13_h3d_hybrid_cnn.pt'
    ):
        super(MultitaskCNNOutput, self).__init__()

        self.num_classes = num_classes
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d())

        self.encoder_seg = nn.Conv2d(512, self.num_classes, 1)
        self.encoder_depth = nn.Conv2d(512, 1, 1)
        self.encoder_ae = nn.Conv2d(512, 3, 1)

        self.score_pool2_seg = nn.Conv2d(16, self.num_classes, 1)
        self.score_pool3_seg = nn.Conv2d(32, self.num_classes, 1)

        self.score_pool2_depth = nn.Conv2d(16, 1, 1)
        self.score_pool3_depth = nn.Conv2d(32, 1, 1)

        self.score_pool2_ae = nn.Conv2d(16, 3, 1)
        self.score_pool3_ae = nn.Conv2d(32, 3, 1)

        self.pretrained = pretrained
        if self.pretrained == True:
            print('Loading CNN weights from %s' % checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location={'cuda:0': 'cpu'})
            self.load_state_dict(checkpoint['model_state'])
            for param in self.parameters():
                param.requires_grad = False
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * (
                        m.out_channels + m.in_channels)
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):

        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)

        encoder_output = self.classifier(conv4)

        encoder_output_seg = self.encoder_seg(encoder_output)
        encoder_output_depth = self.encoder_depth(encoder_output)
        encoder_output_ae = self.encoder_ae(encoder_output)

        score_pool2_seg = self.score_pool2_seg(conv2)
        score_pool3_seg = self.score_pool3_seg(conv3)

        score_pool2_depth = self.score_pool2_depth(conv2)
        score_pool3_depth = self.score_pool3_depth(conv3)

        score_pool2_ae = self.score_pool2_ae(conv2)
        score_pool3_ae = self.score_pool3_ae(conv3)

        score_seg = F.upsample(encoder_output_seg, score_pool3_seg.size()[2:], mode='bilinear')
        score_seg += score_pool3_seg
        score_seg = F.upsample(score_seg, score_pool2_seg.size()[2:], mode='bilinear')
        score_seg += score_pool2_seg
        out_seg = F.upsample(score_seg, x.size()[2:], mode='bilinear')

        score_depth = F.upsample(encoder_output_depth, score_pool3_depth.size()[2:], mode='bilinear')
        score_depth += score_pool3_depth
        score_depth = F.upsample(score_depth, score_pool2_depth.size()[2:], mode='bilinear')
        score_depth += score_pool2_depth
        out_depth = F.sigmoid(F.upsample(score_depth, x.size()[2:], mode='bilinear'))

        score_ae = F.upsample(encoder_output_ae, score_pool3_ae.size()[2:], mode='bilinear')
        score_ae += score_pool3_ae
        score_ae = F.upsample(score_ae, score_pool2_ae.size()[2:], mode='bilinear')
        score_ae += score_pool2_ae
        out_ae = F.sigmoid(F.upsample(score_ae, x.size()[2:], mode='bilinear'))

        return out_seg, out_depth, out_ae

class MultitaskCNN(nn.Module):
    def __init__(
            self,
            num_classes=191,
            pretrained=True,
            checkpoint_path='models/03_13_h3d_hybrid_cnn.pt'
    ):
        super(MultitaskCNN, self).__init__()

        self.num_classes = num_classes
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d())

        self.encoder_seg = nn.Conv2d(512, self.num_classes, 1)
        self.encoder_depth = nn.Conv2d(512, 1, 1)
        self.encoder_ae = nn.Conv2d(512, 3, 1)

        self.score_pool2_seg = nn.Conv2d(16, self.num_classes, 1)
        self.score_pool3_seg = nn.Conv2d(32, self.num_classes, 1)

        self.score_pool2_depth = nn.Conv2d(16, 1, 1)
        self.score_pool3_depth = nn.Conv2d(32, 1, 1)

        self.score_pool2_ae = nn.Conv2d(16, 3, 1)
        self.score_pool3_ae = nn.Conv2d(32, 3, 1)

        self.pretrained = pretrained
        if self.pretrained == True:
            print('Loading CNN weights from %s' % checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location={'cuda:0': 'cpu'})
            self.load_state_dict(checkpoint['model_state'])
            for param in self.parameters():
                param.requires_grad = False
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * (
                        m.out_channels + m.in_channels)
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):

        # assert self.training == False
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)

        return conv4.view(-1, 32 * 10 * 10)




class QuestionLstmEncoder(nn.Module):
    def __init__(self,
                 token_to_idx,
                 wordvec_dim=64,
                 rnn_dim=64,
                 rnn_num_layers=2,
                 rnn_dropout=0):
        super(QuestionLstmEncoder, self).__init__()
        self.token_to_idx = token_to_idx
        self.NULL = token_to_idx['<NULL>']
        self.START = token_to_idx['<START>']
        self.END = token_to_idx['<END>']

        self.embed = nn.Embedding(len(token_to_idx), wordvec_dim)
        self.rnn = nn.LSTM(
            wordvec_dim,
            rnn_dim,
            rnn_num_layers,
            dropout=rnn_dropout,
            batch_first=True)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence
        x_cpu = x.data.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data).long()
        idx = Variable(idx, requires_grad=False)

        hs, _ = self.rnn(self.embed(x))

        idx = idx.view(N, 1, 1).expand(N, 1, hs.size(2))
        H = hs.size(2)
        return hs.gather(1, idx).view(N, H)


# ----------- Act -----------


class ActRnn(nn.Module):
    def __init__(self,
                 image_input=False,
                 image_feat_dim=128,
                 question_input=False,
                 question_embed_dim=128,
                 action_input=False,
                 action_embed_dim=32,
                 position_input=True,
                 position_embed_dim = 32,
                 map_input= True,
                 map_embed_dim = 128,
                 num_actions=10,
                 mode='sl',
                 rnn_type='LSTM',
                 rnn_hidden_dim=128,
                 rnn_num_layers=2,
                 rnn_dropout=0,
                 return_states=False):
        super(ActRnn, self).__init__()

        self.image_input = image_input
        self.image_feat_dim = image_feat_dim

        self.question_input = question_input
        self.question_embed_dim = question_embed_dim

        self.action_input = action_input
        self.action_embed_dim = action_embed_dim

        self.position_input = position_input
        self.position_embed_dim = position_embed_dim

        self.map_input = map_input
        self.map_embed_dim = map_embed_dim

        self.num_actions = num_actions

        self.rnn_type = rnn_type
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers

        self.return_states = return_states

        rnn_input_dim = 0



        if self.position_input == True:
            self.position_embed = nn.Linear(2, position_embed_dim)
            rnn_input_dim += self.position_embed_dim
            print('Adding input to %s: position, rnn dim: %d' % (self.rnn_type,
                                                               rnn_input_dim))

        if self.image_input == True:
            rnn_input_dim += self.image_feat_dim
            print('Adding input to %s: image, rnn dim: %d' % (self.rnn_type,
                                                              rnn_input_dim))

        if self.question_input == True:
            rnn_input_dim += self.question_embed_dim
            print('Adding input to %s: question, rnn dim: %d' %
                  (self.rnn_type, rnn_input_dim))

        if self.action_input == True:

            ##############
            #self.action_embed = nn.Embedding(num_actions, action_embed_dim)
            ##############

            ##############
            self.action_embed = nn.Linear(1, action_embed_dim)
            ##############

            rnn_input_dim += self.action_embed_dim
            print('Adding input to %s: action, rnn dim: %d' % (self.rnn_type,
                                                               rnn_input_dim))

        if self.map_input == True:
            self.map_pool =  nn.AvgPool2d((4,4))
            self.map_embed = nn.Linear(8*8*20,self.map_embed_dim)
            rnn_input_dim += self.map_embed_dim

            print('Adding input to %s: map, rnn dim: %d' % (self.rnn_type,
                                                               rnn_input_dim))




        self.rnn = getattr(nn, self.rnn_type)(
            rnn_input_dim,
            self.rnn_hidden_dim,
            self.rnn_num_layers,
            dropout=rnn_dropout,
            batch_first=True)
        print('Building %s with hidden dim: %d' % (self.rnn_type,
                                                   rnn_hidden_dim))

        self.decoder = nn.Linear(self.rnn_hidden_dim, self.num_actions)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(
                weight.new(self.rnn_num_layers, bsz, self.rnn_hidden_dim)
                .zero_()), Variable(
                    weight.new(self.rnn_num_layers, bsz, self.rnn_hidden_dim)
                    .zero_()))
        elif self.rnn_type == 'GRU':
            return Variable(
                weight.new(self.rnn_num_layers, bsz, self.rnn_hidden_dim)
                .zero_())

    def forward(self,
                img_feats,
                question_feats,
                actions_in,
                position_in,
                map_in,
                action_lengths,
                hidden=False):
        input_feats = Variable()

        T = False

        if self.position_input == True:
            position = self.position_embed(position_in)
            N, T, _ = position.size()
            input_feats = position

        if self.image_input == True:
            if T == False:
                T = actions_in.size(1)
            img_feats = img_feats.repeat(1, T, 1)

            if len(input_feats) == 0:
                input_feats = img_feats
            else:
                input_feats = torch.cat([input_feats, img_feats.float()], 2)
       
        if self.question_input == True:
            N, D = question_feats.size()
            question_feats = question_feats.view(N, 1, D)
            if T == False:
                T = actions_in.size(1)
            question_feats = question_feats.repeat(1, T, 1)
            if len(input_feats) == 0:
                input_feats = question_feats
            else:
                input_feats = torch.cat([input_feats, question_feats.float()], 2)

        if self.action_input == True:
            if len(input_feats) == 0:
                input_feats = self.action_embed(actions_in)
            else:
                input_feats = torch.cat([input_feats, self.action_embed(actions_in.float())], 2)

        if self.map_input == True:
            map_avg = self.map_pool(map_in.float())
            map_avg = map_avg.view(-1,8*8*20)
            map_avg = map_avg.unsqueeze(1)

            if T == False:
                T = actions_in.size(1)
            map_feats = map_avg.repeat(1, T, 1)

            if len(input_feats) == 0:
                input_feats = self.map_embed(map_feats.float())
            else:
    
                input_feats = torch.cat([input_feats, self.map_embed(map_feats.float())], 2)
               
            
        packed_input_feats = pack_padded_sequence(
            input_feats,action_lengths, batch_first=True)
        packed_output, hidden = self.rnn(packed_input_feats)
        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=39)

        output = self.decoder(rnn_output.contiguous().view(
            rnn_output.size(0) * rnn_output.size(1), rnn_output.size(2)))


        if self.return_states == True:
            return rnn_output, output, hidden
        else:
            return output, hidden

    def step_forward(self, img_feats, question_feats, actions_in,position_in,map_in,hidden):
        input_feats = Variable()
        T = False

        if self.position_input == True:
            position = self.position_embed(position_in)
            N, T, _ = position.size()
            input_feats = position

        if self.image_input == True:
            if T == False:
                T = actions_in.size(1)
            img_feats = img_feats.repeat(1, T, 1)

            if len(input_feats) == 0:
                input_feats = img_feats
            else:
                input_feats = torch.cat([input_feats, img_feats.float()], 2)
       
        if self.question_input == True:
            N, D = question_feats.size()
            question_feats = question_feats.view(N, 1, D)
            if T == False:
                T = actions_in.size(1)
            question_feats = question_feats.repeat(1, T, 1)
            if len(input_feats) == 0:
                input_feats = question_feats
            else:
                input_feats = torch.cat([input_feats, question_feats.float()], 2)

        if self.action_input == True:
            if len(input_feats) == 0:
                input_feats = self.action_embed(actions_in)
            else:
                input_feats = torch.cat([input_feats, self.action_embed(actions_in.float())], 2)

        if self.map_input == True:
            map_avg = self.map_pool(map_in.float())
            map_avg = map_avg.view(-1,8*8*20)
            map_avg = map_avg.unsqueeze(1)
            if T == False:
                T = actions_in.size(1)
            map_feats = map_avg.repeat(1, T, 1)

            if len(input_feats) == 0:
                input_feats = self.map_embed(map_feats.float())
            else:
                input_feats = torch.cat([input_feats, self.map_embed(map_feats.float())], 2)

        output, hidden = self.rnn(input_feats, hidden)

        output = self.decoder(output.contiguous().view(
            output.size(0) * output.size(1), output.size(2)))

        return output, hidden




class actPlannerBaseModel(nn.Module):
    def __init__(self,
                 question_vocab,
                 num_output=10,
                 question_wordvec_dim=64,
                 question_hidden_dim=64,
                 question_num_layers=2,
                 question_dropout=0.5,
                 planner_rnn_image_feat_dim=128,
                 planner_rnn_action_embed_dim=32,
                 planner_rnn_position_dim =32,
                 planner_rnn_type='GRU',
                 planner_rnn_hidden_dim=1024,
                 planner_rnn_num_layers=1,
                 planner_rnn_dropout=0):

        super(actPlannerBaseModel, self).__init__()

        self.cnn_fc_layer = nn.Sequential(
            nn.Linear(32 * 10 * 10, planner_rnn_image_feat_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        q_rnn_kwargs = {
            'token_to_idx': question_vocab['questionTokenToIdx'],
            'wordvec_dim': question_wordvec_dim,
            'rnn_dim': question_hidden_dim,
            'rnn_num_layers': question_num_layers,
            'rnn_dropout': question_dropout,
        }

        self.q_rnn = QuestionLstmEncoder(**q_rnn_kwargs)
        self.ques_tr = nn.Sequential(
            nn.Linear(question_hidden_dim, question_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.planner_act_rnn =ActRnn(
            image_input=True,
            image_feat_dim=planner_rnn_image_feat_dim,
            question_input=True,
            question_embed_dim=question_hidden_dim,
            action_input=True,
            action_embed_dim=planner_rnn_action_embed_dim,
            position_input=True,
            position_embed_dim=planner_rnn_position_dim,
            map_input=False,
            num_actions=num_output,
            rnn_type=planner_rnn_type,
            rnn_hidden_dim=planner_rnn_hidden_dim,
            rnn_num_layers=planner_rnn_num_layers,
            rnn_dropout=planner_rnn_dropout,
            return_states=True)


    def forward(self,
                questions,
                planner_img_feats,
                planner_actions_in,
                planner_positions,
                planner_action_lengths,
                planner_hidden_index=None,
                planner_hidden=False):

        # ts = time.time()
        planner_img_feats = self.cnn_fc_layer(planner_img_feats)
        ques_feats = self.q_rnn(questions)
        ques_feats = self.ques_tr(ques_feats)
        planner_states, planner_scores, planner_hidden = self.planner_act_rnn(
            planner_img_feats, ques_feats, planner_actions_in,
            planner_positions,planner_positions,planner_action_lengths)
     
        return planner_scores, planner_hidden

    def planner_step(self, questions, img_feats, actions_in,planner_positions, planner_hidden):

        img_feats = self.cnn_fc_layer(img_feats)
        ques_feats = self.q_rnn(questions)
        ques_feats = self.ques_tr(ques_feats)
        planner_scores, planner_hidden = self.planner_act_rnn.step_forward(
            img_feats, ques_feats, actions_in, planner_positions,planner_positions,planner_hidden)
        return planner_scores, planner_hidden


class actPlannerImproveModel(nn.Module):
    def __init__(self,
                 question_vocab,
                 num_output=10,
                 question_wordvec_dim=64,
                 question_hidden_dim=64,
                 question_num_layers=2,
                 question_dropout=0.5,
                 planner_rnn_image_feat_dim=128,
                 planner_rnn_action_embed_dim=32,
                 planner_rnn_position_dim =32,
                 planner_rnn_map_dim = 128,
                 planner_rnn_type='GRU',
                 planner_rnn_hidden_dim=1024,
                 planner_rnn_num_layers=1,
                 planner_rnn_dropout=0):

        super(actPlannerImproveModel, self).__init__()

        self.cnn_fc_layer = nn.Sequential(
            nn.Linear(32 * 10 * 10, planner_rnn_image_feat_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        q_rnn_kwargs = {
            'token_to_idx': question_vocab['questionTokenToIdx'],
            'wordvec_dim': question_wordvec_dim,
            'rnn_dim': question_hidden_dim,
            'rnn_num_layers': question_num_layers,
            'rnn_dropout': question_dropout,
        }

        self.q_rnn = QuestionLstmEncoder(**q_rnn_kwargs)
        self.ques_tr = nn.Sequential(
            nn.Linear(question_hidden_dim, question_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.planner_act_rnn =ActRnn(
            image_input=True,
            image_feat_dim=planner_rnn_image_feat_dim,
            question_input=True,
            question_embed_dim=question_hidden_dim,
            action_input=True,
            action_embed_dim=planner_rnn_action_embed_dim,
            position_input=True,
            position_embed_dim=planner_rnn_position_dim,
            map_input=True,
            map_embed_dim= planner_rnn_map_dim,
            num_actions=num_output,
            rnn_type=planner_rnn_type,
            rnn_hidden_dim=planner_rnn_hidden_dim,
            rnn_num_layers=planner_rnn_num_layers,
            rnn_dropout=planner_rnn_dropout,
            return_states=True)


    def forward(self,
                questions,
                planner_img_feats,
                planner_actions_in,
                planner_positions,
                planner_map_in,
                planner_action_lengths,
                planner_hidden_index=None,
                planner_hidden=False):

        # ts = time.time()
        planner_img_feats = self.cnn_fc_layer(planner_img_feats)
        ques_feats = self.q_rnn(questions)
        ques_feats = self.ques_tr(ques_feats)
        planner_states, planner_scores, planner_hidden = self.planner_act_rnn(
            planner_img_feats, ques_feats, planner_actions_in,
            planner_positions,planner_map_in,planner_action_lengths)
     
        return planner_scores, planner_hidden

    def planner_step(self, questions, img_feats, actions_in,planner_positions,planner_map_in, planner_hidden):

        img_feats = self.cnn_fc_layer(img_feats)
        ques_feats = self.q_rnn(questions)
        ques_feats = self.ques_tr(ques_feats)
        planner_scores, planner_hidden = self.planner_act_rnn.step_forward(
            img_feats, ques_feats, actions_in, planner_positions,planner_map_in,
            planner_hidden)
        return planner_scores, planner_hidden

class VqaLstmModel(nn.Module):
    def __init__(self,
                 vocab,
                 rnn_wordvec_dim=64,
                 rnn_dim=64,
                 rnn_num_layers=2,
                 rnn_dropout=0.5,
                 fc_use_batchnorm=False,
                 fc_dropout=0.5,
                 fc_dims=(64, )):
        super(VqaLstmModel, self).__init__()
        rnn_kwargs = {
            'token_to_idx': vocab['questionTokenToIdx'],
            'wordvec_dim': rnn_wordvec_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }
        self.rnn = QuestionLstmEncoder(**rnn_kwargs)

        classifier_kwargs = {
            'input_dim': rnn_dim,
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answerTokenToIdx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
            'add_sigmoid': 0
        }
        self.classifier = build_mlp(**classifier_kwargs)

    def forward(self, questions):
        q_feats = self.rnn(questions)
        scores = self.classifier(q_feats)
        return scores


class VqaLstmCnnAttentionModel(nn.Module):
    def __init__(self,
                 vocab,
                 checkpoint_path='models/03_13_h3d_hybrid_cnn.pt',
                 image_feat_dim=64,
                 question_wordvec_dim=64,
                 question_hidden_dim=64,
                 question_num_layers=2,
                 question_dropout=0.5,
                 fc_use_batchnorm=False,
                 fc_dropout=0.5,
                 fc_dims=(64, )):
        super(VqaLstmCnnAttentionModel, self).__init__()

        cnn_kwargs = {'num_classes': 191, 'pretrained': True, 'checkpoint_path':checkpoint_path}
        self.cnn = MultitaskCNN(**cnn_kwargs)
        self.cnn_fc_layer = nn.Sequential(
            nn.Linear(32 * 10 * 10, 64), nn.ReLU(), nn.Dropout(p=0.5))

        q_rnn_kwargs = {
            'token_to_idx': vocab['questionTokenToIdx'],
            'wordvec_dim': question_wordvec_dim,
            'rnn_dim': question_hidden_dim,
            'rnn_num_layers': question_num_layers,
            'rnn_dropout': question_dropout,
        }
        self.q_rnn = QuestionLstmEncoder(**q_rnn_kwargs)

        self.img_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))

        self.ques_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))

        classifier_kwargs = {
            'input_dim': 64,
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answerTokenToIdx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
            'add_sigmoid': 0
        }
        self.classifier = build_mlp(**classifier_kwargs)

        self.att = nn.Sequential(
            nn.Tanh(), nn.Dropout(p=0.5), nn.Linear(128, 1))
        # print('fuck!!')

    def forward(self, images, questions):

        N, T, _, _, _ = images.size()

        # bs x 5 x 3 x 224 x 224
        img_feats = self.cnn(images.contiguous().view(-1, images.size(2), images.size(3), images.size(4)))
        img_feats = self.cnn_fc_layer(img_feats)

        img_feats_tr = self.img_tr(img_feats)

        ques_feats = self.q_rnn(questions)
        ques_feats_repl = ques_feats.view(N, 1, -1).repeat(1, T, 1)
        ques_feats_repl = ques_feats_repl.view(N * T, -1)

        ques_feats_tr = self.ques_tr(ques_feats_repl)

        ques_img_feats = torch.cat([ques_feats_tr, img_feats_tr], 1)

        att_feats = self.att(ques_img_feats)
        att_probs = F.softmax(att_feats.view(N, T), dim=1)
        att_probs2 = att_probs.view(N, T, 1).repeat(1, 1, 64)

        att_img_feats = torch.mul(att_probs2, img_feats.view(N, T, 64))
        att_img_feats = torch.sum(att_img_feats, dim=1)

        mul_feats = torch.mul(ques_feats, att_img_feats)

        scores = self.classifier(mul_feats)

        return scores, att_probs



