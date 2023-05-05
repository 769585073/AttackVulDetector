# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/5/12 14:57
# @Function: define target network

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init

class DetectModel(nn.Module):
    def __init__(self, config):
        super(DetectModel, self).__init__()
        self.activation = F.relu
        self.stop = -1
        self.batch_size = config.batch_size
        self.use_gpu = config.use_gpu
        self.node_list = []
        self.th = torch.cuda if config.use_gpu else torch
        self.batch_node = None
        # self.dropout_embed = nn.Dropout(config.dropout_embed)

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.vocab_size=config.vocab_size
        # pretrained  embedding
        if config.embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(config.embeddings))
            self.embedding.weight.requires_grad = True

        self.hidden_dim = config.lstm_hidden_dim
        self.num_layers = config.lstm_num_layers

        self.C=config.class_num

        # gru,   dropout=config.dropout,
        self.bigru = nn.GRU(config.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.C)
        #  dropout
        self.dropout = nn.Dropout(config.dropout)
        # for name, param in self.bigru.named_parameters():
        #     if name.startswith("weight"):
        #         nn.init.orthogonal_(param)

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def forward(self, x):
        _token_num=500
        token_indexs = [[self.vocab_size-1]*(_token_num-len(item[0]))+item[0] if len(item[0])< _token_num else item[0][:_token_num] for item in x]
        # token_vul_locations = [item[1] if len(item[1])< _token_num else item[1][:900] for item in x]

        # token_indexs=[[self.vocab_size-1]*(_token_num-len(s))+s for s in token_indexs if len(s)<900]
        # token_vul_locations = [[self.vocab_size - 1] * (_token_num - len(s)) + s for s in token_vul_locations if len(s) < 900]
        token_embeds = self.embedding(self.th.LongTensor(token_indexs))

        # embeddings
        input = token_embeds.view(len(x), token_embeds.size(1), -1)
        # gru
        gru_out, _ = self.bigru(input)
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        # gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = torch.tanh(gru_out)
        # drop_out=self.dropout(gru_out)
        # linear
        y = self.hidden2label(gru_out)
        # logit = y
        return y

    def forward_test(self,x):
        _token_num = 500
        token_indexs = [
            [self.vocab_size - 1] * (_token_num - len(item[0])) + item[0] if len(item[0]) < _token_num else item[0][:_token_num]
            for item in x]
        token_embeds = self.embedding(self.th.LongTensor(token_indexs))

        # embeddings
        input = token_embeds.view(len(x), token_embeds.size(1), -1)
        # gru
        gru_out, _ = self.bigru(input)
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = torch.tanh(gru_out)
        # linear
        y = self.hidden2label(gru_out)
        # logit = y
        return input, y

