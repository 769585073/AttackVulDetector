# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/5/15 11:33
# @Function:
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from Utils import Util
from Target_model.RNNDetect import DetectModel
from Utils.mapping import *
from DataProcess.DataPipline import DataPipline
import time
import nltk

class RunNet:
    def __init__(self,config):
        self.config = config
        self.loss_fn=F.cross_entropy

    def train_model(self, model, train_data, optimizer):
        model.train()
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        # train_data = shuffle(train_data)
        while i < len(train_data):
            batch = Util.get_batch(train_data, i, self.config.batch_size)
            i += self.config.batch_size
            train_inputs, train_labels, _, _ = batch
            if self.config.use_gpu:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            output = model(train_inputs)
            # weight = torch.tensor([1, 1.5]).float().cuda() # given different weights for pos and neg class , weight=weight
            loss = self.loss_fn(output, train_labels)
            loss.backward()
            optimizer.step()

            trues.extend(train_labels.cpu().numpy().tolist())
            predicts.extend(torch.max(output, 1)[1].cpu().data.numpy().tolist())
            # calc training acc
            total += len(train_labels)
            total_loss += loss.item() * len(train_inputs)
        train_a, train_p, train_r, train_f = Util.evaluate_b(trues, predicts)
        train_result = (train_a, train_p, train_r, train_f)
        train_loss = total_loss / total
        return train_result, train_loss

    def eval_model(self, model, test_data):
        predicts = []
        trues = []
        program_ids=[]
        total_loss = 0.0
        total = 0.0
        i = 0
        # test_data = shuffle(test_data)
        model.eval()
        while i < len(test_data):
            batch = Util.get_batch(test_data, i, self.config.batch_size)
            i += self.config.batch_size
            test_inputs, test_labels, _, test_program_ids = batch
            if self.config.use_gpu:
                test_inputs, test_labels = test_inputs, test_labels.cuda()

            model.batch_size = len(test_labels)
            output = model(test_inputs)

            loss = self.loss_fn(output, Variable(test_labels))
            # calc valing acc
            # _, predicted = torch.max(output.data, 1)
            predicts.extend(torch.max(output, 1)[1].cpu().data.numpy().tolist())
            trues.extend(test_labels.cpu().numpy().tolist())
            program_ids.extend(test_program_ids)
            total += len(test_labels)
            total_loss += loss.item() * len(test_inputs)
        test_loss = total_loss / total
        test_a, test_p, test_r, test_f = Util.evaluate_b(trues, predicts)
        test_result = (test_a, test_p, test_r, test_f)
        return test_result, test_loss

    def train_eval_model(self,train_data,test_data,model_name,fine_tuning=False):
        train_loss_ = []
        test_loss_ = []
        best_f_measure = 0.0
        train_result_ = []
        test_result_ = []

        detect_model=DetectModel(self.config)
        detect_model.cuda()
        lr = self.config.learning_rate
        if fine_tuning:
            detect_model.load_state_dict(torch.load(self.config.models_path + "VulDetectModel.pt"))
            lr = 6e-6
        parameters = detect_model.parameters()
        # , weight_decay=self.config.weight_decay
        optimizer = torch.optim.Adam(parameters, lr=lr)
        # optimizer = torch.optim.Adamax(parameters, lr=self.config.learning_rate)
        log = open(os.path.join(self.config.log_path, model_name[:-3]+'_log.txt'), 'w')
        test_result, test_loss = self.eval_model(detect_model, test_data)
        test_a, test_p, test_r, test_f = test_result
        print(
            '   Test for iSeVCs: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
                 test_a, test_p, test_r, test_f))
        for epoch in range(self.config.epochs):
            start_time = time.time()
            optimizer = Util.adjust_learning_rate(optimizer, lr, epoch)
            train_result, train_loss = self.train_model(detect_model, train_data, optimizer)
            test_result, test_loss = self.eval_model(detect_model, test_data)

            train_a, train_p, train_r, train_f = train_result
            train_result_.append((train_a, train_p, train_r, train_f))
            train_loss_.append(train_loss)

            test_a, test_p, test_r, test_f = test_result
            test_result_.append((test_a, test_p, test_r, test_f))
            test_loss_.append(test_loss)

            # save model
            if test_f > best_f_measure:
                torch.save(detect_model.state_dict(), self.config.models_path + model_name)
                best_f_measure = test_f

            end_time = time.time()
            print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
                  ' Time Cost: %.3f s'
                  % (epoch + 1, self.config.epochs, train_loss_[epoch], test_loss_[epoch],
                     end_time - start_time))
            log.writelines('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
                  ' Time Cost: %.3f s\n'
                  % (epoch + 1, self.config.epochs, train_loss_[epoch], test_loss_[epoch],
                     end_time - start_time))
            print(
                '   Train: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f], Test for iSeVCs: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
                    train_a, train_p, train_r, train_f, test_a, test_p, test_r, test_f))
            log.writelines(
                '   Train: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f], Test for iSeVCs: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] \n' % (
                    train_a, train_p, train_r, train_f, test_a, test_p, test_r, test_f))
        log.close()
        return True

    def predict_single(self, model, program, norm=False):
        '''
        predicted whether or not program (in program_path) is a vulnerability
        :param model: target model
        :param program:
        :param norm: whether or not performing normalization (transformed into symbolic representation)
        :return:
        '''
        vocab = self.config.vocab
        max_token = self.config.vocab_size-1

        if norm:
            inst_statements = []
            for line in program:
                token_list = create_tokens(line)
                inst_statements.append(token_list)
            map_program, _ = mapping(inst_statements)
        else:
            map_program = program
        input = DataPipline.states2idseq(map_program,vocab,max_token)

        output = model([input])
        predicted = torch.max(output, 1)[1].cpu().data.numpy().tolist()[0] # e.g., predicted: [1], predicted[0]:1
        probability = F.softmax(output,dim=1).cpu().data.numpy().tolist()[0][predicted] # F.softmax : tensor([[0.0026, 0.9974]])
        return predicted, probability