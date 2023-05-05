import copy

import torch
import os
import time
import csv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from gensim.models.word2vec import Word2Vec
from torch.autograd import Variable
from Utils import Util
from Config.ConfigT import MyConf
from Utils.get_tokens import create_tokens
from Utils.mapping import mapping
from DataProcess.DataPipline import DataPipline


class FG(nn.Module):
    def __init__(self, config):
        super(FG, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.vocab_size=config.vocab_size
        if config.embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(config.embeddings))
            self.embedding.weight.requires_grad = True
        self.hidden_dim = config.lstm_hidden_dim
        self.num_layers = config.lstm_num_layers
        self.th = torch.cuda if config.use_gpu else torch
        self.batch_size = config.batch_size

        self.bigru = nn.GRU(config.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        _token_num = 500
        token_indexs = [[self.vocab_size - 1] * (_token_num - len(item[0])) + item[0] if len(item[0]) < _token_num else item[0][:_token_num] for item in x]
        token_embeds = self.embedding(self.th.LongTensor(token_indexs))
        input = token_embeds.view(len(x), token_embeds.size(1), -1)
        gru_out, _ = self.bigru(input)
        gru_out = torch.transpose(gru_out, 1, 2)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = torch.tanh(gru_out)
        return gru_out


class C(nn.Module):
    def __init__(self, config):
        super(C, self).__init__()
        self.C = config.class_num
        self.hidden_dim = config.lstm_hidden_dim
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.C)

    def forward(self, x):
        y = self.hidden2label(x)
        return y


class ZigZagModel(nn.Module):
    def __init__(self, config, FG, C1, C2):
        super(ZigZagModel, self).__init__()
        self.FG = FG
        self.C1 = C1
        self.C2 = C2

    def forward(self, x, train=False):
        gru_out = self.FG(x)
        y1 = self.C1(gru_out)
        y2 = self.C2(gru_out)
        if train:
            return y1, y2
        else:
            return y1


class Train():
    def __init__(self, config):
        self.config = config
        self.loss_fn = F.cross_entropy

    def train_model(self, model, train_data, optimizer, step):
        model.train()
        predicts1 = []
        predicts2 = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0

        self.block_parameters(model, step)
        optimizer = self.get_trainable_parameters(optimizer, model)

        while i<len(train_data):
            batch = Util.get_batch(train_data, i, self.config.batch_size)
            i += self.config.batch_size
            train_inputs, train_labels, _, _ = batch
            curr_frame = train_data.iloc[i: i + self.config.batch_size]
            if self.config.use_gpu:
                train_inputs, train_labels = train_inputs, train_labels.cuda()
            model.zero_grad()
            model.batch_size = len(train_labels)
            output1, output2 = model(train_inputs, train=True)
            probs1 = torch.index_select(output1, 1, train_labels).t()[0].cpu()
            probs2 = torch.index_select(output2, 1, train_labels).t()[0].cpu()
            predicts1.extend(torch.max(output1, 1)[1].cpu().data.numpy().tolist())
            predicts2.extend(torch.max(output2, 1)[1].cpu().data.numpy().tolist())
            if step == 1:
                loss1 = self.loss_fn(output1, train_labels) + self.loss_fn(output2, train_labels)
                loss = loss1
            elif step == 2:
                hard_examples_prob1, hard_examples_prob2 = [], []
                raw_output1, raw_output2, raw_true_labels = [], [], []
                loss1, loss2 = torch.tensor(0.0), torch.tensor(0.0)
                for true_label, prob1, prob2, predict1, predict2, is_adv, o1, o2 in zip(train_labels, probs1, probs2, predicts1, predicts2, curr_frame['is_adv'], output1, output2):
                    if (predict1!=true_label or predict2!=true_label) and is_adv:
                        hard_examples_prob1.append(prob1)
                        hard_examples_prob2.append(prob2)
                    if not is_adv:
                        raw_output1.append(o1)
                        raw_output2.append(o2)
                        raw_true_labels.append(true_label)
                if raw_output1!=[]:
                    loss1 = self.loss_fn(torch.stack(raw_output1), torch.stack(raw_true_labels)) + self.loss_fn(torch.stack(raw_output2), torch.stack(raw_true_labels))
                if hard_examples_prob1!=[]:
                    loss2 = torch.mean(torch.abs(torch.stack(hard_examples_prob1)-torch.stack(hard_examples_prob2)))
                loss1.requires_grad_(True)
                loss2.requires_grad_(True)
                loss = loss1 - loss2
                loss.requires_grad_(True)
            else:
                loss3 = torch.mean(torch.abs(probs1 - probs2))
                loss = loss3
            loss.backward()
            optimizer.step()
            trues.extend(train_labels.cpu().numpy().tolist())
            total += len(train_labels)
            total_loss += loss.item() * len(train_inputs)
        train_a1, train_p1, train_r1, train_f1 = Util.evaluate_b(trues, predicts1)
        train_result1 = (train_a1, train_p1, train_r1, train_f1)
        train_a2, train_p2, train_r2, train_f2 = Util.evaluate_b(trues, predicts2)
        train_result2 = (train_a2, train_p2, train_r2, train_f2)
        train_loss = total_loss / total

        return train_result1, train_result2, train_loss, predicts1, predicts2

    def eval_model(self, model, test_data, step):
        predicts1 = []
        predicts2 = []
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
            curr_frame = test_data.iloc[i: i + self.config.batch_size]

            model.batch_size = len(test_labels)
            output1, output2 = model(test_inputs, train=True)
            probs1 = torch.index_select(output1, 1, test_labels).t()[0].cpu()
            probs2 = torch.index_select(output2, 1, test_labels).t()[0].cpu()

            if step == 1:
                loss1 = self.loss_fn(output1, test_labels) + self.loss_fn(output2, test_labels)
                loss = loss1
            elif step == 2:
                hard_examples_prob1, hard_examples_prob2 = [], []
                raw_output1, raw_output2, raw_true_labels = [], [], []
                loss1, loss2 = torch.tensor(0), torch.tensor(0)
                for true_label, prob1, prob2, predict1, predict2, is_adv, o1, o2 in zip(test_labels, probs1, probs2, predicts1, predicts2, curr_frame['is_adv'], output1, output2):
                    if (predict1!=true_label or predict2!=true_label) and is_adv:
                        hard_examples_prob1.append(prob1)
                        hard_examples_prob2.append(prob2)
                    if not is_adv:
                        raw_output1.append(o1)
                        raw_output2.append(o2)
                        raw_true_labels.append(true_label)
                if raw_output1!=[]:
                    loss1 = self.loss_fn(torch.stack(raw_output1), torch.stack(raw_true_labels)) + self.loss_fn(torch.stack(raw_output2), torch.stack(raw_true_labels))
                if hard_examples_prob1!=[]:
                    loss2 = torch.mean(torch.abs(torch.stack(hard_examples_prob1)-torch.stack(hard_examples_prob2)))
                loss = loss1 - loss2
            else:
                loss3 = torch.mean(torch.abs(probs1 - probs2))
                loss = loss3

            trues.extend(test_labels.cpu().numpy().tolist())
            predicts1.extend(torch.max(output1, 1)[1].cpu().data.numpy().tolist())
            predicts2.extend(torch.max(output2, 1)[1].cpu().data.numpy().tolist())
            program_ids.extend(test_program_ids)
            total += len(test_labels)
            total_loss += loss.item() * len(test_inputs)
        test_loss = total_loss / total
        test_a1, test_p1, test_r1, test_f1 = Util.evaluate_b(trues, predicts1)
        test_result1 = (test_a1, test_p1, test_r1, test_f1)
        test_a1, test_p2, test_r2, test_f2 = Util.evaluate_b(trues, predicts2)
        test_result2 = (test_a1, test_p2, test_r2, test_f2)
        return test_result1, test_result2, test_loss, predicts1, predicts2

    def block_parameters(self, model, step):
        for parameter in model.FG.parameters():
            if step == 3:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True
        for parameter in model.C1.parameters():
            if step == 2:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True
        for parameter in model.C2.parameters():
            if step == 2:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True

    def get_trainable_parameters(self, optimizer, model):
        for param_group in optimizer.param_groups:
            param_group['params'] = list(filter(lambda p: p.requires_grad, model.parameters()))
        return optimizer

    def load_trained_model(self, model_name = 'zig_zag_tmp.pkl'):
        fg = FG(config)
        c1 = C(config)
        c2 = C(config)
        model = ZigZagModel(config, fg, c1, c2)
        if self.config.use_gpu:
            model.cuda()
        model.load_state_dict(torch.load(self.config.models_path + model_name))
        return model

    def train_eval_model(self, raw_train_data, extra_train_data, test_data, model_name, zig_zag_epochs = 25):
        train_loss_ = []
        test_loss_ = []
        best_f_measure = 0.0
        train_result_ = []
        test_result_ = []

        fg = FG(config)
        c1 = C(config)
        c2 = C(config)
        model = ZigZagModel(config, fg, c1, c2)
        if self.config.use_gpu:
            model.cuda()
        lr = self.config.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if not os.path.exists(self.config.models_path + 'zig_zag_tmp.pkl'):
            torch.save(model.state_dict(), self.config.models_path + 'zig_zag_tmp.pkl')
            for epoch in range(zig_zag_epochs):
                start_time1 = time.time()
                optimizer = Util.adjust_learning_rate(optimizer, lr, epoch)

                step = 1
                train_result11, train_result21, train_loss1, predicts11, predicts21 = self.train_model(model, raw_train_data, optimizer, step)
                test_result11, test_result21, test_loss1, test_predicts11, test_predicts21 = self.eval_model(model, test_data, step=step)
                end_time1 = time.time()
                train_a11, train_p11, train_r11, train_f11 = train_result11
                test_a11, test_p11, test_r11, test_f11 = test_result11
                train_a12, train_p12, train_r12, train_f12 = train_result21
                test_a12, test_p12, test_r12, test_f12 = test_result21

                print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
                      ' Time Cost: %.3f s'
                      % (epoch + 1, self.config.epochs, train_loss1, test_loss1,
                         end_time1 - start_time1))
                print(
                    '   1C1Train: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f], 1C1Test for iSeVCs: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
                        train_a11, train_p11, train_r11, train_f11, test_a11, test_p11, test_r11, test_f11))
                print(
                    '   1C2Train: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f], 1C2Test for iSeVCs: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
                        train_a12, train_p12, train_r12, train_f12, test_a12, test_p12, test_r12, test_f12))

                if test_f11 > best_f_measure or test_f12 > best_f_measure:
                    torch.save(model.state_dict(), self.config.models_path + 'zig_zag_tmp.pkl')
                    best_f_measure = test_f11

        model = self.load_trained_model(model_name = 'zig_zag_tmp.pkl')
        best_f_measure = 0.0
        for zig_zag_epoch in range(zig_zag_epochs):
            optimizer = Util.adjust_learning_rate(optimizer, lr, zig_zag_epoch)
            start_time2 = time.time()
            step = 2
            train_data_with_extra = pd.concat((raw_train_data, extra_train_data), axis=0)
            train_data_with_extra = train_data_with_extra.sample(frac=1)
            train_result12, train_result22, train_loss2, predicts12, predictss22 = self.train_model(model, train_data_with_extra, optimizer, step)
            test_result12, test_result22, test_loss2, test_predicts12, test_predicts22 = self.eval_model(model, test_data, step=step)
            end_time2 = time.time()
            train_a21, train_p21, train_r21, train_f21 = train_result12
            test_a21, test_p21, test_r21, test_f21 = test_result12
            train_a22, train_p22, train_r22, train_f22 = train_result22
            test_a22, test_p22, test_r22, test_f22 = test_result22

            print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
                  ' Time Cost: %.3f s'
                  % (zig_zag_epoch + 1, self.config.epochs, train_loss2, test_loss2,
                     end_time2 - start_time2))
            print(
                '   2C1Train: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f], 2C1Test for iSeVCs: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
                    train_a21, train_p21, train_r21, train_f21, test_a21, test_p21, test_r21, test_f21))
            print(
                '   2C2Train: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f], 2C2Test for iSeVCs: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
                    train_a22, train_p22, train_r22, train_f22, test_a22, test_p22, test_r22, test_f22))

            step = 3
            start_time3 = time.time()
            train_result13, train_result23, train_loss3, predicts13, predicts23 = self.train_model(model, extra_train_data, optimizer, step)
            test_result13, test_result23, test_loss3, test_predicts13, test_predicts23 = self.eval_model(model, test_data, step=step)

            end_time3 = time.time()
            train_a31, train_p31, train_r31, train_f31 = train_result13
            test_a31, test_p31, test_r31, test_f31 = test_result13
            train_a32, train_p32, train_r32, train_f32 = train_result23
            test_a32, test_p32, test_r32, test_f32 = test_result23

            print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
                  ' Time Cost: %.3f s'
                  % (zig_zag_epoch + 1, self.config.epochs, train_loss3, test_loss3,
                     end_time3 - start_time3))
            print(
                '   3C1Train: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f], 3C1Test for iSeVCs: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
                    train_a31, train_p31, train_r31, train_f31, test_a31, test_p31, test_r31, test_f31))
            print(
                '   3C2Train: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f], 3C2Test for iSeVCs: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
                    train_a32, train_p32, train_r32, train_f32, test_a32, test_p32, test_r32, test_f32))

            if test_f31 > best_f_measure or test_f32 > best_f_measure:
                torch.save(model.state_dict(), self.config.models_path + model_name)
                best_f_measure = test_f31
        return True


def norm_program(program_slice):
    if program_slice == None:
        return []
    inst_statements = []
    for line in program_slice:
        token_list = create_tokens(line)
        inst_statements.append(token_list)
    map_program, _ = mapping(inst_statements)
    return map_program


def process_slice(slice):
    split_token = '<EOL>'
    slice_statements = slice.split(split_token)
    statements = []
    for statement in slice_statements:
        if not statement.endswith('\n'):
            statement = statement+'\n'
        statements.append(statement)
    return statements


def get_extra_examples(config):
    tool = DataPipline(config)
    file_names = []
    for file in os.listdir(config.result_path):
        file_names.append(file)
    ori_train_data = pd.read_pickle(config.data_path + 'train/blocks.pkl')
    columns = ['data_id', 'SyVCs', 'file_fun', 'program_id', 'types', 'map_code', 'orig_code', 'label', 'token_indexs_length', 'is_adv']
    data_ids, SyVCs, files, program_ids, types, map_code_slices, orig_code_slices, labels, token_indexs_length, is_adv = [], [], [], [], [], [], [], [], [], []
    for file_name in file_names:
        with open(os.path.join(config.result_path, file_name), 'r') as csv_f:
            reader = csv.reader(csv_f)
            content = list(reader)
            for line in content[1:1001]:
                index, ground_label, ori_label, ori_prob, ori_slice, adv_label, adv_prob, adv_slice, query_times = line

                adv_slice = process_slice(adv_slice)
                adv_slice_norm = norm_program(adv_slice)

                data_ids.append(ori_train_data.loc[int(index)]['data_id'])
                SyVCs.append(ori_train_data.loc[int(index)]['SyVCs'])
                files.append(ori_train_data.loc[int(index)]['file_fun'])
                program_ids.append(ori_train_data.loc[int(index)]['program_id'])
                types.append(ori_train_data.loc[int(index)]['types'])
                map_code_slices.append(adv_slice_norm)
                orig_code_slices.append(adv_slice)
                labels.append(int(ground_label))
                if config.Norm_symbol:
                    token_indexs_length.append(tool.states2idseq(adv_slice_norm, config.vocab, config.vocab_size - 1))
                else:
                    token_indexs_length.append(tool.states2idseq(adv_slice, config.vocab, config.vocab_size - 1))
                is_adv.append(True)
    raw_retrain_data_path = os.path.join(config.defence_path, 'raw_retrain_data.pkl')
    is_adv_raw_data = [False for _ in range(len(ori_train_data))]
    ori_train_data['is_adv'] = is_adv_raw_data
    ori_train_data.to_pickle(raw_retrain_data_path)
    data = {'data_id': data_ids, 'SyVCs': SyVCs, 'file_fun': files, 'program_id': program_ids, 'types': types, 'map_code': map_code_slices, 'orig_code': orig_code_slices,
            'label': labels, 'token_indexs_length': token_indexs_length, 'is_adv': is_adv}
    adv_data = pd.DataFrame(data, columns=columns)
    adv_data_path = os.path.join(config.defence_path, 'adv_data.pkl')
    adv_data.to_pickle(adv_data_path)
    return ori_train_data, adv_data


def main(config):
    word2vec = Word2Vec.load(config.embedding_path + "/node_w2v_60").wv
    config.embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    config.embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    config.embedding_dim = word2vec.vectors.shape[1]
    config.vocab_size = word2vec.vectors.shape[0] + 1
    config.vocab = word2vec.vocab

    # raw_train_data = pd.read_pickle(config.data_path + 'train/blocks.pkl')
    # train_data_with_extra = pd.read_pickle(os.path.join(config.defence_path, 'adv_train_data_1.0.pkl'))
    test_data_path = os.path.join(config.defence_path, 'test_retrain_data.pkl')
    if os.path.exists(test_data_path):
        test_data = pd.read_pickle(test_data_path)
    else:
        test_data_ori = pd.read_pickle(config.data_path + 'test/blocks.pkl')
        is_adv_test_data = [False for _ in range(len(test_data_ori))]
        test_data_ori['is_adv'] = is_adv_test_data
        test_data_ori.to_pickle(test_data_path)
        test_data = test_data_ori

    adv_data_path = os.path.join(config.defence_path, 'adv_data.pkl')
    raw_retrain_data_path = os.path.join(config.defence_path, 'raw_retrain_data.pkl')
    if os.path.exists(adv_data_path) and os.path.exists(raw_retrain_data_path):
        raw_train_data = pd.read_pickle(raw_retrain_data_path)
        extra_train_data = pd.read_pickle(adv_data_path)
    else:
        raw_train_data, extra_train_data = get_extra_examples(config)

    zig_zag_epochs = 25
    model_name = 'vul_detect_zigzag.pt'
    train = Train(config)
    # train.train_eval_model(raw_train_data, extra_train_data, test_data, model_name=model_name, zig_zag_epochs = zig_zag_epochs)
    model = train.load_trained_model(model_name='zig_zag_tmp.pkl')
    # model = train.load_trained_model(model_name=model_name)
    results1, results2, _, _, _ =train.eval_model(model, test_data, step=1)
    print(results1)
    print(results2)


if __name__ == '__main__':
    config = MyConf('../Config/config_defence.cfg')
    main(config)