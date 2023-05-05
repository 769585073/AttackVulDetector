# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/5/14 10:22
# @Function: Generating adversarial program samples
import os
import torch
from Utils.mapping import *
from DataProcess.DataPipline import DataPipline
from Target_model.RNNDetect import DetectModel
import numpy as np

import torch.nn.functional as F


class AttackTarget:
    def __init__(self, config):
        self.config = config
        self.max_token = self.config.vocab_size - 1
        self.num_candida_tok=5 # the number of candidates used to replace the original word
        self.prob_fn = F.softmax
        self.detect_model=None

    def load_trained_model(self,model_name):
        '''
        load pretrained target model
        :return:
        '''
        if self.detect_model!=None:
            return self.detect_model
        else:
            detect_model = DetectModel(self.config)
            if self.config.use_gpu:
                detect_model.cuda()
            # load model
            if os.path.exists(self.config.models_path + model_name):
                # 预训练critic model
                detect_model.load_state_dict(torch.load(self.config.models_path + model_name))
                self.detect_model=detect_model
                return self.detect_model
            else:
                print('No Pretrained Model, Please Train first!')
                return None

    def generate_adver_samples(self, program, norm=False):
        '''
        :param program: a list of statements, each of which is a string
        :param norm: whether or not performing normalization (transformed into symbolic representation)
        :return:
        '''
        if norm:
            inst_statements = []
            for line in program:
                token_list = create_tokens(line)
                inst_statements.append(token_list)
            map_program, _ = mapping(inst_statements)
        else:
            map_program= program

        token_sequence, s_lengths = DataPipline.states2idseq(map_program,self.config.vocab, self.max_token)
        detect_model=self.load_trained_model('VulDetectModel.pt')
        detect_model.zero_grad()

        token_features, score = detect_model.forward_test([[token_sequence, s_lengths]]) # token_features are the embedding of token_sequences
        token_features.retain_grad() # non-leaf nodes, should setting the attribute before performing backward and then accessing their gradient

        self.find_best_token_bugger(map_program,token_features, s_lengths, score)
        # self.statement_impact(map_program, score)

    def token_impact_improve(self,token_features,s_lengths, score):
        '''
        not finished (21-5-25)
        :param token_features:
        :param s_lengths:
        :param score:
        :return:
        '''
        token_len = sum(s_lengths)
        # token_features: [1, 500, token embedding sizes], 500 is the maximum number of token sequences for each program
        # x_features=token_features[0][500-token_len:]
        gradient_x = []
        for token_index in range(token_len):
            gradient_x.append(
                token_features.grad[0][500 - token_len + token_index].sum().item())  # gradient of each word in x
        self.detect_model.zero_grad()
        opposite_predicted_label = torch.min(score, 1)[1].cpu().data.numpy().tolist()[0]  # e.g., predicted: [1], predicted[0]:1
        # torch.ones_like(score)
        score[0][opposite_predicted_label].backward()


        # w_order = sorted(token_grad_dict.items(), key=lambda x: x[1], reverse=True)
        gradient_x_a = np.array(gradient_x)
        sort_index = np.argsort(-gradient_x_a)
        return sort_index, gradient_x


    def statement_impact(self, map_program, orig_score):
        '''
        compute the importance score of each statement according to query target model
        :param map_program:
        :param orig_score:
        :return:
        '''
        state_score_lst = []
        orig_positive_score=orig_score[0][1].item() # obtain score value of original program
        for state_index in range(len(map_program)):
            _temp_map_program=[map_program[i] for i in range(len(map_program)) if i != state_index]
            # predicting program without the state_index-th statement
            token_sequence, s_lengths = DataPipline.states2idseq(_temp_map_program, self.config.vocab, self.max_token)
            detect_model = self.load_trained_model('VulDetectModel.pt')
            _, score = detect_model.forward_test([[token_sequence, s_lengths]])  # token_features are the embedding of token_sequences
            state_score = orig_positive_score-score[0][1].item()
            state_score_lst.append(state_score)

        state_score_arr = np.array(state_score_lst)
        sort_index = np.argsort(-state_score_arr)
        print(sort_index)

    def token_impact(self,token_features, s_lengths, score):
        '''
        compute the importance score of each token according to the gradient of output
        :param token_features:
        :param s_lengths:
        :param score:
        :return:
        '''
        predicted_label = torch.max(score, 1)[1].cpu().data.numpy().tolist()[0]  # e.g., predicted: [1], predicted[0]:1
        # torch.ones_like(score)
        # score[0][predicted_label].backward()
        # torch.tensor([[-1,1]])
        score.backward(torch.tensor([[1,0]]).cuda())  # after performing this statement, check the gradients of token features
        # torch.autograd.backward(score,grad_tensors=torch.ones_like(score))

        # the number of tokens in token_sequence, each element in s_lengths is the number of tokens in each statement
        token_len = sum(s_lengths)
        # token_features: [1, 500, token embedding sizes], 500 is the maximum number of token sequences for each program
        # x_features=token_features[0][500-token_len:]
        gradient_x = []
        # token_grad_dict = {}
        for token_index in range(token_len):
            gradient_x.append(
                token_features.grad[0][500 - token_len + token_index].sum().item())  # gradient of each word in x
            # token_grad_dict[token_sequence[i]] = gradient_x[i]  # word symbol: gradient value

        # w_order = sorted(token_grad_dict.items(), key=lambda x: x[1], reverse=True)
        gradient_x_a = np.array(gradient_x)
        sort_index = np.argsort(-gradient_x_a)
        return sort_index, gradient_x

    def find_best_token_bugger(self,map_program,token_features,s_lengths,orig_score):
        # sort tokens according to their importance (sort_index), importance score (gradient_x)
        sort_index, gradient_x = self.token_impact(token_features, s_lengths, orig_score)

        token_symbol_sequence = []
        [token_symbol_sequence.extend(create_tokens(statement)) for statement in map_program]
        #
        for token_index in sort_index:
            # if self.num_candida_tok == 5:
            #     print('try: %s, Not Found'% total)
            #     return 'Not Found!'
            # replace token with new tokens (i.e., bugger generated in this function)
            orig_token = token_symbol_sequence[token_index]
            # i in which statement
            statement, line_num, token_line_index = self.token_in_line(token_index, token_symbol_sequence, s_lengths)
            print('Statement: %s, line_num: %s, token_line_index: %s' % (statement, line_num, token_line_index))
            # mutation for the statement



    def token_in_line(self,token_index, token_sequences, s_lengths):
        '''
        obtain the statement and the number of line of code, to which the token_i belong
        :param token_index:
        :param token_sequences:
        :param s_lengths:
        :return:
        '''
        sum_len=0
        for line_num, len in enumerate(s_lengths):
            start_index=sum_len
            sum_len+=len
            if token_index<sum_len and token_index>=start_index:
                token_line_index=token_index-start_index # token index in the line, to which the token_i belong
                return token_sequences[start_index:sum_len], line_num, token_line_index



