import copy
import json
import os
import csv
import re
import random
import string
import time

import torch

import numpy as np
import torch.nn.functional as F

from Utils.mapping import mapping
from Utils.get_tokens import create_tokens
from DataProcess.DataPipline import DataPipline
from Target_model.RNNDetect import DetectModel
from CParser.ParseAndMutCode import ParseAndMutCode
from collections import OrderedDict
from Utils.Util import concat_statement
from Defence.zigzag_defence import ZigZagModel, FG, C


class Obfuscation():
    def __init__(self, config, replace_limit_number, sample_ids, model_name='VulDetectModel.pt'):
        self.config = config
        self.model_name = model_name
        self.detect_model = None
        self.detect_model = self.load_trained_model(model_name)
        self.max_token = self.config.vocab_size - 1
        self.vocab = self.config.vocab
        self.replace_limit_number = replace_limit_number
        self.sample_ids = sample_ids
        self.check_model_num = 0

    def load_trained_model(self, model_name):
        '''
        load pretrained target model
        :return:
        '''
        if self.detect_model!=None:
            return self.detect_model
        else:
            if model_name=='vul_detect_zigzag.pt':
                fg = FG(self.config)
                c1 = C(self.config)
                c2 = C(self.config)
                detect_model = ZigZagModel(self.config, fg, c1, c2)
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

    def make_batch(self, map_programs, batch_size=256, norm = False):
        dataset = []
        batch = []
        for i, program in enumerate(map_programs):
            if i % batch_size == 0 and batch != []:
                dataset.append(batch)
                batch = []
            if norm:
                map_program = self.norm_program(program)
            else:
                map_program = program
            input_seq = DataPipline.states2idseq(map_program, self.vocab, self.max_token)
            batch.append(input_seq)
        if batch != []:
            dataset.append(batch)
        return dataset

    def run_model_by_batch(self, dataloader, model=None):
        # 按照batch返回对应的label和该label下的概率
        if model == None:
            model = self.detect_model
        predicted = []
        probability = []
        for batch in dataloader:
            model.batch_size = len(batch)
            logits = model(batch)
            softs = F.softmax(logits, dim=1)
            preds = torch.max(softs, 1)[1]
            probs = torch.index_select(softs, 1, preds).t()[0]
            predicted.extend(preds.cpu().data.numpy().tolist())
            probability.extend(probs.cpu().data.numpy().tolist())
            self.check_model_num += 1
        return predicted, probability

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

    def norm_program(self, program_slice):
        # 标准化切片
        inst_statements = []
        for line in program_slice:
            token_list = create_tokens(line)
            inst_statements.append(token_list)
        map_program, _ = mapping(inst_statements)
        return map_program

    def predict_adv_program(self, adv_program):
        # 预测单个切片
        self.check_model_num += 1
        vocab = self.config.vocab
        max_token = self.config.vocab_size-1
        input = DataPipline.states2idseq(adv_program,vocab,max_token)
        output = self.detect_model([input])
        predicted = torch.max(output, 1)[1].detach().cpu().data.numpy().tolist()[0]
        probability = F.softmax(output,dim=1).detach().cpu().data.numpy().tolist()[0][predicted]
        return predicted, probability

    def find_best_token_bugger(self,map_program,token_features,s_lengths, orig_score):
        # sort tokens according to their importance (sort_index), importance score (gradient_x)
        # 梯度方法
        sort_index, gradient_x = self.token_impact(token_features, s_lengths, orig_score)
        token_symbol_sequence = []
        [token_symbol_sequence.extend(create_tokens(statement)) for statement in map_program]
        ori_label, ori_prob = self.predict_adv_program(map_program)
        diff = 0
        adv_program = map_program
        for token_index in sort_index:
            # if self.num_candida_tok == 5:
            #     print('try: %s, Not Found'% total)
            #     return 'Not Found!'
            # replace token with new tokens (i.e., bugger generated in this function)
            orig_token = token_symbol_sequence[token_index]
            if 'variable' in orig_token or 'func' in orig_token:
                continue
            # i in which statement
            statement, line_num, token_line_index = self.token_in_line(token_index, token_symbol_sequence, s_lengths)
            # print('Statement: %s, line_num: %s, token_line_index: %s' % (statement, line_num, token_line_index))
            # mutation for the statement
            # 在这修改 改后若对抗成功则返回对抗样本
            adv_program = self.macro_replace(adv_program, line_num, token_line_index, diff, diff)
            diff += 1
            # adv_program = self.macroReplace(map_program, line_num, token_line_index, diff, program_slice)
            adv_label, adv_prob = self.predict_adv_program(adv_program)
            if adv_label != ori_label:
                return True, adv_program, diff
            if diff >= self.replace_limit_number:
                break
        return False, adv_program, diff

    def black_box(self, slice_program, map_program, s_lengths):
        # 查模型方法 宏替换
        ori_label, ori_prob = self.predict_adv_program(map_program)
        token_symbol_sequence = []
        [token_symbol_sequence.extend(create_tokens(statement)) for statement in slice_program]
        sort_index = self.token_impact_blackbox(ori_label, ori_prob, s_lengths, token_symbol_sequence)
        diff = 0
        best_adv_program = slice_program
        best_adv_norm = map_program
        best_prob = 1.0
        for token_index in sort_index:
            # orig_token = token_symbol_sequence[token_index]
            # if 'variable' in orig_token or 'func' in orig_token:
            #     continue
            statement, line_num, token_line_index = self.token_in_line(token_index, token_symbol_sequence, s_lengths)
            adv_program = self.macro_replace(slice_program, line_num, token_line_index, diff, ori_label, diff, method=1)
            adv_norm = self.norm_program(adv_program)
            diff += 1
            adv_label, adv_prob = self.predict_adv_program(adv_norm)
            if adv_label != ori_label:
                return True, adv_program, adv_norm, diff
            else:
                if adv_prob < best_prob:
                    best_prob = adv_prob
                    best_adv_program = adv_program
                    best_adv_norm = adv_norm
        return False, best_adv_program, best_adv_norm, diff

    def macro_replace(self, slice_program, line_num, token_line_index, t, ori_label, pos, method=2):
        if method == 0:
            macro = 'unknown_' + str(t)
        elif method == 1:
            random.seed(time.time())
            macro = ''.join(random.choice(string.ascii_uppercase) for i in range(5))
        else:
            best_word = self.get_best_word(slice_program, ori_label, pos)
            macro = best_word.upper()
        adv_program = copy.deepcopy(slice_program)
        replace_line_list = create_tokens(adv_program[line_num])
        token_line_index = min(token_line_index, len(replace_line_list)-1)
        replace_line_list[token_line_index] = macro
        replace_line_content = " ".join(replace_line_list)
        adv_program[line_num] = replace_line_content
        return adv_program

    def generate_adver_samples(self, program_slice, map_program):
        '''
        :param program: a list of statements, each of which is a string
        :param norm: whether or not performing normalization (transformed into symbolic representation)
        :return:
        '''
        token_sequence, s_lengths = DataPipline.states2idseq(program_slice, self.config.vocab, self.max_token)
        detect_model = self.detect_model
        detect_model.zero_grad()

        token_features, score = detect_model.forward_test([[token_sequence, s_lengths]]) # token_features are the embedding of token_sequences
        token_features.retain_grad() # non-leaf nodes, should setting the attribute before performing backward and then accessing their gradient

        # 梯度 宏替换
        # is_sucess, adv_program, diff = self.find_best_token_bugger(map_program, token_features, s_lengths, score)

        # 查模型 宏替换
        is_sucess, adv_program, obf_norm, diff = self.black_box(program_slice, map_program, s_lengths)

        return is_sucess, adv_program, obf_norm, diff

    def replace_lines_adver_samples(self, slice_program, map_program, exchange_dict, search):
        # exchange_dict保存每一行与其可交换行的字典 行号是切片中的行号
        diff = 1
        ori_label, ori_prob = self.predict_adv_program(map_program)
        if search == 0:
            sort_index = [random.choice([i for i in range(len(slice_program))])]
        elif search == 1:
            sort_index = self.statement_impact(map_program, ori_label, ori_prob)[:1]
        else:
            sort_index = self.statement_impact(map_program, ori_label, ori_prob)
        min_prob = ori_prob
        best_adv = slice_program
        slice_program_t = copy.deepcopy(slice_program)
        best_obf = copy.deepcopy(map_program)
        for line_index in sort_index:
            if str(line_index) in exchange_dict.keys():
                # for down_index in exchange_dict[str(line_index)]:
                down_index = int(exchange_dict[str(line_index)])
                if down_index >= len(map_program):
                    continue
                map_program[line_index], map_program[down_index] = map_program[down_index], map_program[line_index]
                slice_program_t[line_index], slice_program_t[down_index] = slice_program_t[down_index], slice_program_t[line_index]
                adv_label, adv_prob = self.predict_adv_program(map_program)
                # print(ori_prob, adv_prob)
                if adv_label!=ori_label:
                    return True, slice_program_t, map_program, diff
                if adv_prob <= min_prob:
                    min_prob = adv_prob
                    best_adv = copy.deepcopy(slice_program_t)
                    best_obf = copy.deepcopy(map_program)
                map_program[line_index], map_program[down_index] = map_program[down_index], map_program[line_index]
                slice_program_t[line_index], slice_program_t[down_index] = slice_program_t[down_index], slice_program_t[line_index]
        return False, best_adv, best_obf, diff

    def get_variable_impact(self, map_program, total_variable, best_line, best_word, ori_label, ori_prob):
        variable_score_dict = {}
        for variable_declaration in total_variable:
            # insert_statement_dead_code = 'if(' + variable_declaration + '==' + variable_declaration + ')\n'
            insert_statement_dead_code = 'if(false){' + variable_declaration + '=' + '\"' + best_word + '\" ;}'
            map_insert = self.norm_program([insert_statement_dead_code])
            _temp_map_program = map_program[:best_line+1] + map_insert + map_program[best_line+1:]

            label, prob = self.predict_adv_program(_temp_map_program)
            if label!=ori_label:
                state_score = 1
            else:
                state_score = ori_prob-prob
            variable_score_dict[variable_declaration] = state_score
        return variable_score_dict

    def insert_statement1(self, variable_declaration, best_word):
        # line1 = 'if(false){' + variable_declaration + '=' + '\"101\" ;}'
        # line2 = 'if(false){' + variable_declaration + '=' + '\"' + best_word + '\" ;}'
        # line3 = 'if(false){' + variable_declaration + '=' + '\"51\" ;}'
        # line1 = 'printf(\"' + best_word + ', %p' + '\", &' + variable_declaration + ');'
        # line2 = 'printf(\"' + best_word + ', %p' + '\", &' + variable_declaration + ');'
        # line3 = 'printf(\"' + best_word + ', %p' + '\", &' + variable_declaration + ');'
        # line1 = 'printf(\"' + best_word + ', %p' + '\", &' + variable_declaration + ');'
        # line1 = 'if(false){(int) ' + variable_declaration + '=0;}'
        # line3 = 'if(false){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line1 = "int " + best_word + ";"
        # return [line1, line2, line3]

        line1 = 'printf(\"' + best_word + ', %p=' + '101\", &' + variable_declaration + ');'
        # line1 = 'printf(\"' + best_word + ', %p' + '\", &' + variable_declaration + ');'
        # line1 = 'while(false){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line1 = 'if(false){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line1 = 'if(false){(int) ' + variable_declaration + '=0;}'
        # line1 = 'if(' + variable_declaration + '!=' + variable_declaration + '){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        return [line1]

    def insert_statement0(self, variable_declaration, best_word):
        # 在无漏洞的代码中插入漏洞语句会造成0变1的攻击成功率提升
        # line1 = 'if(' + best_word + ' != ' + best_word + '){ char * void[50] ;'
        # line2 = variable_declaration + '=' + '\"' + best_word + '\" ;'
        # line3 = 'void[51] = &' + variable_declaration + ' ;}'
        # line1 = 'if(' + variable_declaration + '!=' + variable_declaration + '){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line2 = 'if(' + variable_declaration + '!=' + variable_declaration + '){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line3 = 'if(' + variable_declaration + '!=' + variable_declaration + '){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line1 = 'if(' + variable_declaration + '!=' + variable_declaration + '){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line2 = 'while(false){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line3 = 'if(false){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line1 = 'if(false){(int) ' + variable_declaration + '=0;}'
        # line1 = "int " + best_word + ";"
        # return [line1, line2, line3]

        # line1 = 'printf(\"' + best_word + ', %p=' + '100\", &' + variable_declaration + ');'
        line1 = 'printf(\"' + best_word + ', %p' + '\", &' + variable_declaration + ');'
        # line1 = 'while(false){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line1 = 'if(false){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        # line1 = 'if(false){(int) ' + variable_declaration + '=0;}'
        # line1 = 'if(' + variable_declaration + '!=' + variable_declaration + '){(char *) &' + variable_declaration + '=\"' + best_word + '\";}'
        return [line1]

    def token_importance(self, ori_label, ori_prob, s_lengths, token_symbol_sequence):
        map_programs = []
        for token_index in range(len(token_symbol_sequence)):
            tmp_token_list = []
            for i in range(len(token_symbol_sequence)):
                if i != token_index:
                    tmp_token_list.append(token_symbol_sequence[i])
                else:
                    tmp_token_list.append("")
            tmp_map_program = []
            start = 0
            for l in s_lengths:
                tmp_map_program.append(" ".join(tmp_token_list[start:start+l]))
                start += l
            map_programs.append(tmp_map_program)
        dataloader = self.make_batch(map_programs, norm=True)
        preds, probs = self.get_label_prob(dataloader, ori_label)

        state_score_arr = np.array(probs)
        sort_index = np.argsort(state_score_arr)
        p = []
        for i in sort_index:
            p.append(probs[i])
        return sort_index, p

    def get_label_prob(self, dataloader, ori_label, model=None):
        if model == None:
            model = self.detect_model
        predicted = []
        probability = []
        for batch in dataloader:
            model.batch_size = len(batch)
            logits = model(batch)
            softs = F.softmax(logits, dim=1)
            preds = torch.max(softs, 1)[1]
            if ori_label == 0:
                tmp = torch.zeros_like(preds)
            else:
                tmp = torch.ones_like(preds)
            probs = torch.index_select(softs, 1, tmp).t()[0]
            predicted.extend(preds.cpu().data.numpy().tolist())
            predicted.extend(preds.cpu().data.numpy().tolist())
            probability.extend(probs.cpu().data.numpy().tolist())
        return predicted, probability

    def create_best_word_json(self):
        # test_data = pd.read_pickle(self.config.data_path + 'test/blocks.pkl')
        token_importance_dict = {0: {}, 1: {}}
        count = 0
        # for index, program in test_data.iterrows():
        for index in self.sample_ids:
            count += 1
            print(count, index)
            with open(os.path.join(self.config.sample_slice_path, str(index)), 'r') as f:
                slice = f.readlines()
            # slice = program['orig_code']
            map = self.norm_program(slice)
            ori_label_new, ori_prob = self.predict_adv_program(map)
            token_symbol_sequence = []
            [token_symbol_sequence.extend(create_tokens(statement)) for statement in slice]
            _, s_lengths = DataPipline.states2idseq(slice, self.config.vocab, self.config.vocab_size - 1)
            sort_index, probs = self.token_importance(ori_label_new, ori_prob, s_lengths, token_symbol_sequence)
            for p in range(2):
                i = sort_index[p]
                prob = probs[p]
                if token_symbol_sequence[i] in token_importance_dict[ori_label_new].keys():
                    # 按变化概率确定词的评分
                    token_importance_dict[ori_label_new][token_symbol_sequence[i]] = max(token_importance_dict[ori_label_new][token_symbol_sequence[i]], ori_prob - prob)
                else:
                    token_importance_dict[ori_label_new][token_symbol_sequence[i]] = ori_prob - prob
        with open(os.path.join(self.config.temp_path, "token_importance_" + self.model_name + ".json"), 'w') as token_importance_jsn:
            json.dump(token_importance_dict, token_importance_jsn)

    def get_best_word(self, slice_program, ori_label, pos):
        if not os.path.exists(os.path.join(self.config.temp_path, "token_importance_" +self.model_name+ ".json")):
            self.create_best_word_json()
        with open(os.path.join(self.config.temp_path, "token_importance_" +self.model_name+ ".json"), 'r') as token_importance_jsn:
            token_importance_dict = json.load(token_importance_jsn)
        curr_tokens = []
        for sentence in slice_program:
            curr_tokens.extend(create_tokens(sentence))
        tmp = token_importance_dict[str(1-ori_label)]
        tmp_list = []
        for token in tmp.keys():
            tmp_list.append([token, int(tmp[token])])
        tmp_list.sort(key=lambda x: int(x[1]), reverse=True)
        ret = []
        count = 10
        for t in tmp_list:
            if t[0] in curr_tokens:
                continue
            ret.append([t[0]])
            count -= 1
            if count <= 0:
                break
        best_word = ret[pos][0]
        return best_word

    def add_print(self, slice_program, map_program, add_dict):
        # 在切片上添加打印语句和死代码
        diff = 0
        ori_label, ori_prob = self.predict_adv_program(map_program)
        sort_index = self.statement_impact(map_program, ori_label, ori_prob)
        # insert_line = -1
        insert_line_list = []

        # 选取当前评分最低的语句
        # best_word_index = sort_index[-1]
        # best_word = slice_program[best_word_index]

        # 选择对当前类别评分最低的token
        # best_word = self.get_best_word([], ori_label, 0)
        best_word = ''.join(random.choice(string.ascii_uppercase) for i in range(5))

        # 选择能够插入的行
        best_adv_slice = copy.deepcopy(slice_program)
        best_adv_prob = ori_prob
        for line_index in sort_index:
            if line_index == 0 or str(line_index-1) not in add_dict.keys():
                continue
            if add_dict[str(line_index-1)] == []:
                continue
            # insert_line = line_index
            # break
            insert_line_list.append(line_index)
        # if insert_line > 0:
        for insert_line in insert_line_list:
            variable_declaration = random.choice(add_dict[str(insert_line - 1)])
            if ori_label == 0:
                insert_statement_dead_code = self.insert_statement0(variable_declaration, best_word)
            else:
                insert_statement_dead_code = self.insert_statement1(variable_declaration, best_word)
            adv_slice = slice_program[:insert_line] + insert_statement_dead_code + slice_program[insert_line:]
            adv_program = self.norm_program(adv_slice)
            diff += 1
            adv_label, adv_prob = self.predict_adv_program(adv_program)
            if adv_label != ori_label:
                return True, adv_slice, diff, adv_label, adv_prob
            else:
                if adv_prob < best_adv_prob:
                    best_adv_prob = adv_prob
                    best_adv_slice = adv_slice
        return False, best_adv_slice, diff, ori_label, best_adv_prob

    def statement_impact(self, map_program, ori_label, ori_prob):
        '''
        compute the importance score of each statement according to query target model
        :param map_program:
        :param orig_score:
        :return:
        '''
        state_score_lst = []
        # insert_statement = ['printf("%x\\n", &' + 'variable' + ');\n']
        # map_insert_statement = self._normProgram(insert_statement)
        # orig_positive_score=orig_score[0][1].item() # obtain score value of original program
        for state_index in range(len(map_program)):
            # _temp_map_program=[map_program[i] for i in range(len(map_program)) if i != state_index]
            # predicting program without the state_index-th statement
            # token_sequence, s_lengths = DataPipline.states2idseq(_temp_map_program, self.config.vocab, self.max_token)
            # _, score = self.detect_model.forward_test([[token_sequence, s_lengths]])  # token_features are the embedding of token_sequences
            # _temp_map_program = map_program[:state_index+1] + map_insert_statement + map_program[state_index+1:]
            _temp_map_program = map_program[:state_index] + map_program[state_index + 1:]

            label, prob = self.predict_adv_program(_temp_map_program)
            if label!=ori_label:
                state_score = 1
            else:
                state_score = ori_prob-prob
            state_score_lst.append(state_score)

        state_score_arr = np.array(state_score_lst)
        # print(state_score_arr)
        sort_index = np.argsort(-state_score_arr)
        return sort_index

    def token_impact_blackbox(self, ori_label, ori_prob, statement_lenths, token_symbol_sequence):
        token_score_list = []
        map_programs = []
        for token_index in range(len(token_symbol_sequence)):
            tmp_token_list = []
            for i in range(len(token_symbol_sequence)):
                if i != token_index:
                    tmp_token_list.append(token_symbol_sequence[i])
                else:
                    # tmp_token_list.append("unknown_")
                    tmp_token_list.append("")
            tmp_map_program = []
            start = 0
            for l in statement_lenths:
                tmp_map_program.append(" ".join(tmp_token_list[start:start+l]))
                start += l
            map_programs.append(tmp_map_program)
        dataloader = self.make_batch(map_programs, norm=True)
        preds, probs = self.run_model_by_batch(dataloader)

        i = 0
        while i < len(preds):
            label = preds[i]
            score = probs[i]
            if label != ori_label:
                state_score = ori_prob - (1-score)
            else:
                state_score = ori_prob - score
            token_score_list.append(state_score)
            i += 1

        state_score_arr = np.array(token_score_list)
        sort_index = np.argsort(-state_score_arr)
        return sort_index

    def macro_raplace_attack(self):
        index_label_path = os.path.join(self.config.temp_path, 'index_label.json')
        with open(index_label_path, 'r') as index_label_jsn:
            index_label_dict = json.load(index_label_jsn)

        sample_path = self.config.sample_path
        total, sucess = 0, 0
        result_path = os.path.join(self.config.result_path, 'macro_attack_'+self.model_name+'_results.csv')
        with open(result_path, 'w', encoding='utf-8', newline='') as r_csv_f:
            csv_writer = csv.writer(r_csv_f)
            csv_writer.writerow(['index', 'true_label', 'ori_label', 'ori_prob', 'ori_code', 'adv_label', 'adv_prob', 'adv_code', 'query_times'])
            for index in self.sample_ids:
                self.check_model_num = 0
                index = str(index)
                slice_program_path = os.path.join(self.config.sample_slice_path, index)
                with open(slice_program_path, 'r') as slice_program_f:
                    slice_program = slice_program_f.readlines()
                # with open(os.path.join(sample_path, index), "r") as ori_code_f:
                #     map_program = ori_code_f.readlines()
                map_program = self.norm_program(slice_program)
                original_label, original_probability = self.predict_adv_program(map_program)
                if original_label!=int(index_label_dict[index]):
                    slice_program = concat_statement(slice_program)
                    csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, original_label, original_probability, slice_program, self.check_model_num])
                    continue
                is_sucess, obf, obf_norm, diff = self.generate_adver_samples(slice_program, map_program)
                adv_label, adv_probability = self.predict_adv_program(obf_norm)
                total += 1
                if is_sucess:
                    sucess += 1
                print(sucess, total, sucess/total)
                slice_program = concat_statement(slice_program)
                obf = concat_statement(obf)
                csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, adv_label, adv_probability, obf, self.check_model_num])
            csv_writer.writerow([sucess, total, sucess/total])

    def create_exchange_dict(self):
        sample_source_path = self.config.sample_source_path
        temp_path = self.config.temp_path

        if not os.path.exists(os.path.join(temp_path, 'filename_exchange_line.json')):
            print("processing source code and get filename_exchange_line.json!")
            pm = ParseAndMutCode()
            pm.translate_c_exchange_line(sample_source_path, temp_path)

        index_id_path = os.path.join(temp_path, 'index_id.json')
        with open(index_id_path, 'r') as index_id_jsn:
            index_id_dict = json.load(index_id_jsn)

        index_line_number_path = os.path.join(temp_path, 'index_line_number.json')
        with open(index_line_number_path, 'r') as index_line_number_jsn:
            index_line_number_dict = json.load(index_line_number_jsn)

        filename_exchange_line_path = os.path.join(temp_path, 'filename_exchange_line.json')
        with open(filename_exchange_line_path, 'r') as filename_exchange_line_jsn:
            filename_exchange_line_dict = json.load(filename_exchange_line_jsn)

        count = 0
        index_exchange_dict = {}

        for index in self.sample_ids:
            print(index, count)
            count += 1

            index = str(index)
            id = index_id_dict[index]

            program_slice_path = os.path.join(self.config.sample_slice_path, index)
            with open(program_slice_path, 'r') as program_slice_f:
                program_slice = program_slice_f.readlines()

            line_dict = index_line_number_dict[index]
            exchange_dict = {}
            for ori_line in range(len(program_slice) - 1):
                down_line = ori_line + 1
                ori_line = str(ori_line)
                down_line = str(down_line)
                # 没有匹配到行号的切片行忽略掉
                if -1 in line_dict[ori_line] or -1 in line_dict[down_line] or None in line_dict[down_line] or None in \
                        line_dict[down_line]:
                    continue
                # 切片中相邻的两行在源文件中不相邻 或者 两行不在同一个文件中 行号有不对的
                try:
                    if int(line_dict[ori_line][0]) != int(line_dict[down_line][0]) - 1 or line_dict[ori_line][1] != \
                            line_dict[down_line][1]:
                        continue
                except:
                    continue
                code = program_slice[int(ori_line)].split()
                down_code = program_slice[int(down_line)].split()
                negative_keywords = ['for', 'while', 'dowhile', 'if', 'else', 'switch', 'case', 'default', 'continue', 'break', 'return']
                if any(s in code for s in negative_keywords) or any(s in down_code for s in negative_keywords):
                    continue

                key = id + '/' + line_dict[ori_line][1]
                if key not in filename_exchange_line_dict.keys() or str(line_dict[ori_line][0]) not in filename_exchange_line_dict[key].keys():
                    continue
                if filename_exchange_line_dict[key][str(line_dict[ori_line][0])]!='-1':
                    exchange_dict[ori_line] = down_line
            index_exchange_dict[index] = exchange_dict

        index_exchange_line_path = os.path.join(temp_path, 'index_exchange_line_dict.json')
        with open(index_exchange_line_path, 'w') as index_exchange_line_jsn:
            json.dump(index_exchange_dict, index_exchange_line_jsn)

    def replace_line_attack(self):
        sample_path = self.config.sample_path
        temp_path = self.config.temp_path
        search = 2
        if search == 0:
            search_m = "random"
        elif search == 1:
            search_m = "greedy"
        else:
            search_m = "all"

        if not os.path.exists(os.path.join(temp_path, 'index_exchange_line_dict.json')):
            print("creating exchange_line_dict!")
            self.create_exchange_dict()
        with open(os.path.join(temp_path, 'index_exchange_line_dict.json'), 'r') as index_exchange_line_jsn:
            index_exchange_line_dict = json.load(index_exchange_line_jsn)

        index_label_path = os.path.join(self.config.temp_path, 'index_label.json')
        with open(index_label_path, 'r') as index_label_jsn:
            index_label_dict = json.load(index_label_jsn)

        count = 0
        sucess = 0
        result_path = os.path.join(self.config.result_path, 'exchange_line_attack_'+self.model_name+'_results('+search_m+').csv')
        with open(result_path, 'w', encoding='utf-8', newline='') as r_csv_f:
            csv_writer = csv.writer(r_csv_f)
            csv_writer.writerow(['index', 'true_label', 'ori_label', 'ori_prob', 'ori_code', 'adv_label', 'adv_prob', 'adv_code', 'query_times'])
            for index in self.sample_ids:
                self.check_model_num = 0
                index = str(index)
                # map_program_path = os.path.join(sample_path, index)
                # with open(map_program_path, 'r') as map_program_f:
                #     map_program = map_program_f.readlines()

                slice_program_path = os.path.join(self.config.sample_slice_path, index)
                with open(slice_program_path, 'r') as slice_program_f:
                    slice_program = slice_program_f.readlines()
                map_program = self.norm_program(slice_program)

                original_label, original_probability = self.predict_adv_program(map_program)
                if original_label!=int(index_label_dict[index]):
                    slice_program = concat_statement(slice_program)
                    csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, original_label, original_probability, slice_program, self.check_model_num])
                    continue
                count += 1
                exchange_dict = index_exchange_line_dict[index]
                is_sucess, adv_slice, obf, diff = self.replace_lines_adver_samples(slice_program, map_program, exchange_dict, search)
                if is_sucess:
                    sucess += 1
                adv_label, adv_probability = self.predict_adv_program(obf)
                print(sucess, count, sucess/count)
                slice_program = concat_statement(slice_program)
                adv_slice = concat_statement(adv_slice)
                csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, adv_label, adv_probability, adv_slice, self.check_model_num])
            csv_writer.writerow([sucess, count, sucess/count])

    def create_add_dict(self):
        # 获取每一行的变量声明{index：{line_number：variable_declarations}}形式
        sample_source_path = self.config.sample_source_path
        temp_path = self.config.temp_path

        if not os.path.exists(os.path.join(temp_path, 'filename_outspace.json')):
            print("processing source code and get filename_outspace.json!")
            pm = ParseAndMutCode()
            pm.translate_c_add_print(sample_source_path, temp_path)

        index_id_path = os.path.join(temp_path, 'index_id.json')
        with open(index_id_path, 'r') as index_id_jsn:
            index_id_dict = json.load(index_id_jsn)

        index_line_number_path = os.path.join(temp_path, 'index_line_number.json')
        with open(index_line_number_path, 'r') as index_line_number_jsn:
            index_line_number_dict = json.load(index_line_number_jsn)

        filename_outspace_path = os.path.join(temp_path, 'filename_outspace.json')
        with open(filename_outspace_path, 'r') as filename_outspace_jsn:
            filename_outspace_dict = json.load(filename_outspace_jsn)

        index_add_dict = {}
        count = 0
        for index in self.sample_ids:
            print(index, count)
            count += 1

            index = str(index)
            id = index_id_dict[index]
            tmp_line_dict = index_line_number_dict[index]

            # 让字典中的key即切片行号按照从小到大的顺序放在字典里
            tmp = []
            for t in tmp_line_dict.keys():
                tmp.append([int(t), tmp_line_dict[t]])
            tmp.sort(key=lambda x: x[0])
            line_dict = OrderedDict()
            for t in tmp:
                line_dict[t[0]] = t[1]

            add_dict = OrderedDict()
            pre = set()

            for slice_line in line_dict.keys():
                if -1 in line_dict[slice_line] or None in line_dict[slice_line]:
                    continue
                try:
                    # 切片中有的行号不对
                    ori_line = str(line_dict[slice_line][0])
                except:
                    continue
                key = id + '/' + line_dict[slice_line][1]
                if key not in filename_outspace_dict.keys() or ori_line not in filename_outspace_dict[key].keys():
                    continue
                variable_declaration = set(filename_outspace_dict[key][ori_line])
                if variable_declaration != set():
                    # 去除数组声明的中括号 去除函数括号
                    variable_declaration_list = [re.sub(u"\\(.*?\\)|\\[.*?]|\\{.*?}", "", x) for x in variable_declaration]
                    variable_declaration = set(variable_declaration_list).union(pre)
                    add_dict[slice_line] = list(variable_declaration)
                    pre = set(variable_declaration)
                else:
                    add_dict[slice_line] = list(pre)
            index_add_dict[index] = add_dict

        index_add_dict_path = os.path.join(temp_path, 'index_add_dict.json')
        with open(index_add_dict_path, 'w') as index_add_dict_jsn:
            json.dump(index_add_dict, index_add_dict_jsn)

    def add_dead_code_attack(self):
        sample_path = self.config.sample_path
        temp_path = self.config.temp_path
        sample_slice_path = self.config.sample_slice_path

        if not os.path.exists(os.path.join(temp_path, 'index_add_dict.json')):
            print("creating add_dict!")
            self.create_add_dict()
        with open(os.path.join(temp_path, 'index_add_dict.json'), 'r') as index_add_dict_jsn:
            index_add_dict = json.load(index_add_dict_jsn)

        index_label_path = os.path.join(self.config.temp_path, 'index_label.json')
        with open(index_label_path, 'r') as index_label_jsn:
            index_label_dict = json.load(index_label_jsn)

        count = 0
        sucess = 0
        result_path = os.path.join(self.config.result_path, 'add_deadcode_attack_' + self.model_name + '_results(random_1, all_location, 5).csv')
        with open(result_path, 'w', encoding='utf-8', newline='') as r_csv_f:
            csv_writer = csv.writer(r_csv_f)
            csv_writer.writerow(
                ['index', 'true_label', 'ori_label', 'ori_prob', 'ori_code', 'adv_label', 'adv_prob', 'adv_code', 'query_times'])
            for index in self.sample_ids:
                self.check_model_num = 0
                index = str(index)

                slice_program_path = os.path.join(sample_slice_path, index)
                with open(slice_program_path, 'r') as slice_program_f:
                    slice_program = slice_program_f.readlines()

                # map_program_path = os.path.join(sample_path, index)
                # with open(map_program_path, 'r') as map_program_f:
                #     map_program = map_program_f.readlines()
                map_program = self.norm_program(slice_program)
                original_label, original_probability = self.predict_adv_program(map_program)
                if original_label!=int(index_label_dict[index]):
                    slice_program = concat_statement(slice_program)
                    csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, original_label, original_probability, slice_program, self.check_model_num])
                    continue
                add_dict = index_add_dict[index]
                count += 1
                is_sucess, obf, diff, adv_label, adv_probability = self.add_print(slice_program, map_program, add_dict)
                slice_program = concat_statement(slice_program)
                obf = concat_statement(obf)
                csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, adv_label, adv_probability, obf, self.check_model_num])
                if is_sucess:
                    sucess += 1
                print(sucess, count, sucess/count)
            csv_writer.writerow([sucess, count, sucess/count])

    def replace_const(self, program_slice, map_program, replace_dict):
        # 常量替换生成对抗样本
        diff = 1
        ori_label, ori_prob = self.predict_adv_program(map_program)
        min_prob = ori_prob
        best_adv = map_program
        best_slice = program_slice
        for ori_line in replace_dict.keys():
            replace_content = replace_dict[ori_line]
            ori_line = int(ori_line)
            adv_slice = program_slice[0:ori_line+1] + replace_content + program_slice[ori_line+2:]
            adv_map = self.norm_program(adv_slice)
            adv_label, adv_prob = self.predict_adv_program(adv_map)
            if adv_label != ori_label:
                return True, adv_slice, adv_map, diff
            if adv_prob <= min_prob:
                min_prob = adv_prob
                best_adv = copy.deepcopy(adv_map)
                best_slice = copy.deepcopy(adv_slice)
        return False, best_slice, best_adv, diff

    def create_replace_dict(self):
        # 获取每一行的变量声明{index：{line_number：replace_content}}形式
        sample_source_path = self.config.sample_source_path
        temp_path = self.config.temp_path

        if not os.path.exists(os.path.join(temp_path, 'filename_replace_const.json')):
            print("processing source code and get filename_replace_const.json!")
            pm = ParseAndMutCode()
            pm.translate_c_replace_const(sample_source_path, temp_path)

        index_id_path = os.path.join(temp_path, 'index_id.json')
        with open(index_id_path, 'r') as index_id_jsn:
            index_id_dict = json.load(index_id_jsn)

        index_line_number_path = os.path.join(temp_path, 'index_line_number.json')
        with open(index_line_number_path, 'r') as index_line_number_jsn:
            index_line_number_dict = json.load(index_line_number_jsn)

        filename_replace_const_path = os.path.join(temp_path, 'filename_replace_const.json')
        with open(filename_replace_const_path, 'r') as filename_replace_const_jsn:
            filename_replace_const_dict = json.load(filename_replace_const_jsn)

        index_replace_dict = {}
        count = 0
        for index in self.sample_ids:
            print(index, count)
            count += 1

            index = str(index)
            id = index_id_dict[index]
            line_dict = index_line_number_dict[index]

            program_slice_path = os.path.join(self.config.sample_slice_path, index)
            with open(program_slice_path, 'r') as program_slice_f:
                program_slice = program_slice_f.readlines()
            line_replace_dict = {}
            for slice_line in range(len(program_slice)):
                slice_line = str(slice_line)
                if -1 in line_dict[slice_line] or None in line_dict[slice_line]:
                    continue
                try:
                    # 切片中有的行号不对
                    ori_line = str(line_dict[slice_line][0])
                except:
                    continue
                key = id + '/' + line_dict[slice_line][1]
                if key in filename_replace_const_dict.keys() and ori_line in filename_replace_const_dict[key].keys():
                    line_replace_dict[slice_line] = filename_replace_const_dict[key][ori_line]
            index_replace_dict[index] = line_replace_dict

        index_replace_dict_path = os.path.join(temp_path, 'index_replace_dict.json')
        with open(index_replace_dict_path, 'w') as index_replace_dict_jsn:
            json.dump(index_replace_dict, index_replace_dict_jsn)

    def replace_const_attack(self):
        sample_path = self.config.sample_path
        temp_path = self.config.temp_path
        sample_slice_path = self.config.sample_slice_path

        if not os.path.exists(os.path.join(temp_path, 'index_replace_dict.json')):
            print("creating index_replace_dict!")
            self.create_replace_dict()
        with open(os.path.join(temp_path, 'index_replace_dict.json'), 'r') as index_replace_dict_jsn:
            index_replace_dict = json.load(index_replace_dict_jsn)

        index_label_path = os.path.join(self.config.temp_path, 'index_label.json')
        with open(index_label_path, 'r') as index_label_jsn:
            index_label_dict = json.load(index_label_jsn)

        count = 0
        sucess = 0
        result_path = os.path.join(self.config.result_path, 'replace_const_attack_'+self.model_name+'_results.csv')
        with open(result_path, 'w', encoding='utf-8', newline='') as r_csv_f:
            csv_writer = csv.writer(r_csv_f)
            csv_writer.writerow(
                ['index', 'true_label', 'ori_label', 'ori_prob', 'ori_code', 'adv_label', 'adv_prob', 'adv_code',
                 'query_times'])
            for index in self.sample_ids:
                self.check_model_num = 0
                index = str(index)

                slice_program_path = os.path.join(sample_slice_path, index)
                with open(slice_program_path, 'r') as slice_program_f:
                    slice_program = slice_program_f.readlines()

                # map_program_path = os.path.join(sample_path, index)
                # with open(map_program_path, 'r') as map_program_f:
                #     map_program = map_program_f.readlines()
                map_program = self.norm_program(slice_program)
                original_label, original_probability = self.predict_adv_program(map_program)
                if original_label!=int(index_label_dict[index]):
                    slice_program = concat_statement(slice_program)
                    csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, original_label, original_probability, slice_program, self.check_model_num])
                    continue
                line_replace_dict = index_replace_dict[index]
                count += 1
                is_sucess, adv_slice, obf, diff = self.replace_const(slice_program, map_program, line_replace_dict)
                adv_label, adv_probability = self.predict_adv_program(obf)
                slice_program = concat_statement(slice_program)
                adv_slice = concat_statement(adv_slice)
                csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, adv_label,adv_probability, adv_slice, self.check_model_num])
                if is_sucess:
                    sucess += 1
                print(sucess, count, sucess / count)
            csv_writer.writerow([sucess, count, sucess / count])

    def create_merge_line_dict(self):
        # 获取函数合并可删除的函数调用和函数声明的行{index：{line:stopline}}
        sample_source_path = self.config.sample_source_path
        temp_path = self.config.temp_path

        if not os.path.exists(os.path.join(temp_path, 'filename_merge_function.json')):
            print("processing source code and get filename_merge_function.json!")
            pm = ParseAndMutCode()
            pm.translate_c_merge_function(sample_source_path, temp_path)

        index_id_path = os.path.join(temp_path, 'index_id.json')
        with open(index_id_path, 'r') as index_id_jsn:
            index_id_dict = json.load(index_id_jsn)

        index_line_number_path = os.path.join(temp_path, 'index_line_number.json')
        with open(index_line_number_path, 'r') as index_line_number_jsn:
            index_line_number_dict = json.load(index_line_number_jsn)

        filename_merge_function_path = os.path.join(temp_path, 'filename_merge_function.json')
        with open(filename_merge_function_path, 'r') as filename_merge_function_jsn:
            filename_merge_function_dict = json.load(filename_merge_function_jsn)

        index_merge_dict = {}
        count = 0
        for index in self.sample_ids:
            print(index, count)
            count += 1

            index = str(index)
            id = index_id_dict[index]
            line_dict = index_line_number_dict[index]

            program_slice_path = os.path.join(self.config.sample_slice_path, index)
            with open(program_slice_path, 'r') as program_slice_f:
                program_slice = program_slice_f.readlines()

            line_stop_dict = {}
            for slice_line in range(len(program_slice)):
                slice_line = str(slice_line)
                if -1 in line_dict[slice_line] or None in line_dict[slice_line]:
                    continue
                try:
                    # 切片中有的行号不对
                    ori_line = str(line_dict[slice_line][0])
                except:
                    continue

                key = id + '/' + line_dict[slice_line][1]
                if key in filename_merge_function_dict.keys() and ori_line in filename_merge_function_dict[key].keys():
                    definitaion_line = filename_merge_function_dict[key][ori_line][0]
                    definitaion_file = filename_merge_function_dict[key][ori_line][1]
                    if int(slice_line)+1 < len(program_slice):
                        if -1 in line_dict[slice_line] or None in line_dict[slice_line]:
                            line_stop_dict[slice_line] = slice_line
                            continue
                        try:
                            down_line = line_dict[slice_line][0]
                            down_line_file = line_dict[slice_line][1]
                            if down_line == definitaion_line and down_line_file == definitaion_file:
                                line_stop_dict[slice_line] = down_line
                            else:
                                line_stop_dict[slice_line] = slice_line
                        except:
                            line_stop_dict[slice_line] = slice_line
            index_merge_dict[index] = line_stop_dict
        index_merge_dict_path = os.path.join(temp_path, 'index_merge_dict.json')
        with open(index_merge_dict_path, 'w') as index_merge_dict_jsn:
            json.dump(index_merge_dict, index_merge_dict_jsn)

    def merge_function(self, program_slice, map_program, line_merge_dict):
        # 删除函数调用的语句行与 展开的函数声明行
        diff = 1
        ori_label, ori_prob = self.predict_adv_program(map_program)
        min_prob = ori_prob
        adv_slice = program_slice
        best_adv = map_program
        for start_line in line_merge_dict.keys():
            end_line = line_merge_dict[start_line]
            start_line, end_line = int(start_line), int(end_line)
            adv_slice = program_slice[:start_line] + program_slice[end_line+1:]
            adv_map = self.norm_program(adv_slice)
            adv_label, adv_prob = self.predict_adv_program(adv_map)
            if adv_label != ori_label:
                return True, adv_slice, adv_map, diff
            if adv_prob <= min_prob:
                min_prob = adv_prob
                best_adv = copy.deepcopy(adv_map)
        return False, adv_slice, best_adv, diff

    def merge_function_attack(self):
        sample_path = self.config.sample_path
        temp_path = self.config.temp_path
        sample_slice_path = self.config.sample_slice_path

        if not os.path.exists(os.path.join(temp_path, 'index_merge_dict.json')):
            print("creating index_merge_dict!")
            self.create_merge_line_dict()
        with open(os.path.join(temp_path, 'index_merge_dict.json'), 'r') as index_merge_line_dict_jsn:
            index_merge_line_dict = json.load(index_merge_line_dict_jsn)

        index_label_path = os.path.join(self.config.temp_path, 'index_label.json')
        with open(index_label_path, 'r') as index_label_jsn:
            index_label_dict = json.load(index_label_jsn)

        count = 0
        sucess = 0
        result_path = os.path.join(self.config.result_path, 'merge_function_attack_'+self.model_name+'_results.csv')
        with open(result_path, 'w', encoding='utf-8', newline='') as r_csv_f:
            csv_writer = csv.writer(r_csv_f)
            csv_writer.writerow(
                ['index', 'true_label', 'ori_label', 'ori_prob', 'ori_code', 'adv_label', 'adv_prob', 'adv_code', 'query_times'])
            for index in self.sample_ids:
                self.check_model_num = 0
                index = str(index)

                slice_program_path = os.path.join(sample_slice_path, index)
                with open(slice_program_path, 'r') as slice_program_f:
                    slice_program = slice_program_f.readlines()

                map_program_path = os.path.join(sample_path, index)
                # with open(map_program_path, 'r', encoding='utf-8') as map_program_f:
                #     map_program = map_program_f.readlines()
                map_program = self.norm_program(slice_program)

                line_merge_dict = index_merge_line_dict[index]
                original_label, original_probability = self.predict_adv_program(map_program)
                if original_label!=int(index_label_dict[index]):
                    slice_program = concat_statement(slice_program)
                    csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, original_label, original_probability, slice_program, self.check_model_num])
                    continue
                count += 1
                is_sucess, adv_slice, obf, diff = self.merge_function(slice_program, map_program, line_merge_dict)
                adv_label, adv_probability = self.predict_adv_program(obf)
                slice_program = concat_statement(slice_program)
                adv_slice = concat_statement(adv_slice)
                csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, adv_label, adv_probability, adv_slice, self.check_model_num])
                if is_sucess:
                    sucess += 1
                print(sucess, count, sucess / count)
            csv_writer.writerow([sucess, count, sucess / count])

    def unroll_loop_attack(self):
        sample_path = self.config.sample_path
        sample_slice_path = self.config.sample_slice_path

        index_label_path = os.path.join(self.config.temp_path, 'index_label.json')
        with open(index_label_path, 'r') as index_label_jsn:
            index_label_dict = json.load(index_label_jsn)

        count = 0
        sucess = 0
        result_path = os.path.join(self.config.result_path, 'unroll_loop_attack_'+self.model_name+'_results.csv')
        with open(result_path, 'w', encoding='utf-8', newline='') as r_csv_f:
            csv_writer = csv.writer(r_csv_f)
            csv_writer.writerow(['index', 'true_label', 'ori_label', 'ori_prob', 'ori_code', 'adv_label', 'adv_prob', 'adv_code', 'query_times'])
            for index in self.sample_ids:
                self.check_model_num = 0
                index = str(index)

                slice_program_path = os.path.join(sample_slice_path, index)
                with open(slice_program_path, 'r') as slice_program_f:
                    slice_program = slice_program_f.readlines()

                map_program_path = os.path.join(sample_path, index)
                with open(map_program_path, 'r', encoding='utf-8') as map_program_f:
                    map_program = map_program_f.readlines()

                original_label, original_probability = self.predict_adv_program(map_program)
                if original_label != int(index_label_dict[index]):
                    slice_program = concat_statement(slice_program)
                    csv_writer.writerow([index, index_label_dict[index], original_label, original_probability, slice_program, original_label, original_probability, slice_program,
                                         self.check_model_num])
                    continue
                count += 1
                is_sucess, adv_slice, obf, diff = self.unroll_loop(slice_program, map_program)
                adv_label, adv_probability = self.predict_adv_program(obf)
                slice_program = concat_statement(slice_program)
                adv_slice = concat_statement(adv_slice)
                csv_writer.writerow(
                    [index, index_label_dict[index], original_label, original_probability, slice_program, adv_label, adv_probability, adv_slice, self.check_model_num])
                if is_sucess:
                    sucess += 1
                print(sucess, count, sucess / count)
            csv_writer.writerow([sucess, count, sucess / count])

    def unroll_loop(self, slice_program, map_program):
        diff = 0
        ori_label, ori_prob = self.predict_adv_program(map_program)
        min_prob = ori_prob
        adv_slice = slice_program
        best_adv = map_program
        best_word = ''.join(random.choice(string.ascii_uppercase) for i in range(5))
        for i, line in enumerate(slice_program):
            # best_word = self.get_best_word(slice_program, ori_label, pos=1)
            line_tokens = create_tokens(line)
            if "while" in line_tokens:
                r1 = re.compile(r'[(](.*)[)]', re.S)
                exp = re.findall(r1, line)
                # statements = ["while(1){\n", "if(!("+ exp[0] + "))break;\n"] # 19/913
                # statements= ["int "+best_word+" =1\n", "while("+best_word+"){\n", "if(!("+exp[0]+"))"+best_word+"=0;\n"] # 19/913
                # statements = ["while(1){\n", "if(!("+ exp[0] + "))break;\n", "else continue;\n"]
                # statements = ["int " + best_word + " =1\n", "while(" + best_word + "){\n", "if(!(" + exp[0] + "))" + best_word + "=0;\n", "else continue;\n"] # 14/913
                statements = ["bool " + best_word + " =true\n", "while(" + best_word + "){\n", "if(!(" + exp[0] + "))" + best_word + "=false;\n"]
                # statements = ["bool " + best_word + " =true\n", "while(" + best_word + "){\n", "if(!(" + exp[0] + "))" + best_word + "=false;\n", "else continue;\n"]
                adv_slice = slice_program[:i] + statements +  slice_program[i+1:]
                adv_map = self.norm_program(adv_slice)
                adv_label, adv_prob = self.predict_adv_program(adv_map)
                if adv_label != ori_label:
                    diff += 1
                    return True, adv_slice, adv_map, diff
                if adv_prob <= min_prob:
                    min_prob = adv_prob
                    best_adv = copy.deepcopy(adv_map)
                    diff += 1
            elif "for" in line_tokens:
                r1 = re.compile(r'[(](.*)[)]', re.S)
                exp = re.findall(r1, line)
                i_s = exp[0].split(';')
                if len(i_s)!=3:
                    continue
                # print(line)
                # print(i_s)
                # statements = ["for("+i_s[0]+';;'+i_s[2]+")\n", "if(!("+ i_s[1] + "))break;\n"]
                # statements = ["int " + best_word + " =1\n", "for("+i_s[0]+';;'+i_s[2]+")\n", "if(!("+i_s[1]+"))"+best_word+"=0;\n"]
                # statements = ["for("+i_s[0]+';;'+i_s[2]+")\n", "if(!("+ i_s[1] + "))break;\n", "else continue;\n"]
                # statements = ["int " + best_word + " =1\n", "for("+i_s[0]+';;'+i_s[2]+")\n", "if(!("+i_s[1]+"))"+best_word+"=0;\n", "else continue;\n"]
                statements = ["bool " + best_word + " =true\n", "for("+i_s[0]+';'+best_word+';'+i_s[2]+")\n", "if(!("+i_s[1]+"))"+best_word+"=false;\n"]
                # statements = ["bool " + best_word + " =true\n", "for(" + i_s[0] + ';;' + i_s[2] + ")\n", "if(!(" + i_s[1] + "))" + best_word + "=false;\n", "else continue;\n"]
                adv_slice = slice_program[:i] + statements +  slice_program[i+1:]
                adv_map = self.norm_program(adv_slice)
                adv_label, adv_prob = self.predict_adv_program(adv_map)
                if adv_label != ori_label:
                    diff += 1
                    return True, adv_slice, adv_map, diff
                if adv_prob <= min_prob:
                    min_prob = adv_prob
                    best_adv = copy.deepcopy(adv_map)
                    diff += 1


            # 直接把bool变量替换写在这了 攻击成功率0.0
            # if "true" in line_tokens:
            #     statements = [line.replace("true", '"' + best_word + '" == "' + best_word + '"')]
            #     adv_slice = slice_program[:i] + statements +  slice_program[i+1:]
            #     adv_map = self.norm_program(adv_slice)
            #     adv_label, adv_prob = self.predict_adv_program(adv_map)
            #     if adv_label != ori_label:
            #         diff += 1
            #         return True, adv_slice, adv_map, diff
            #     if adv_prob <= min_prob:
            #         min_prob = adv_prob
            #         best_adv = copy.deepcopy(adv_map)
            # if "false" in line:
            #     statements = [line.replace("false", '"' + best_word + '" != "' + best_word + '"')]
            #     adv_slice = slice_program[:i] + statements +  slice_program[i+1:]
            #     adv_map = self.norm_program(adv_slice)
            #     adv_label, adv_prob = self.predict_adv_program(adv_map)
            #     if adv_label != ori_label:
            #         diff += 1
            #         return True, adv_slice, adv_map, diff
            #     if adv_prob <= min_prob:
            #         min_prob = adv_prob
            #         best_adv = copy.deepcopy(adv_map)

        return False, adv_slice, best_adv, diff

