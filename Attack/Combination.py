import copy
import json
import os
import csv
import random
import time
import string
import re

import numpy as np
import pandas as pd

from Attack.Mutation import Mutation
from Attack.Obfuscation import Obfuscation
from Utils.get_tokens import create_tokens
from DataProcess.DataPipline import DataPipline
from Utils.nope import NopeTool
from Utils.Util import concat_statement


class Combination():
    def __init__(self, config, replace_limit_number, sample_ids, model_name = 'VulDetectModel.pt'):
        self.config = config
        self.check_model_num = 0
        self.replace_limit_number = replace_limit_number
        self.sample_ids = sample_ids
        self.obfuscation = Obfuscation(config, replace_limit_number, sample_ids, model_name)
        self.mutation = Mutation(config, sample_ids, self.obfuscation.detect_model, model_name)
        self.load_temp_files()
        self.token_list = None
        test_data = pd.read_pickle(self.config.data_path + 'test/blocks.pkl')
        self.nope_tool = NopeTool(self.config, test_data)
        self.model_name = model_name

    def load_temp_files(self):
        if not os.path.exists(os.path.join(self.config.temp_path, "index_add_dict.json")):
            self.obfuscation.create_add_dict()
        with open(os.path.join(self.config.temp_path, "index_add_dict.json"), 'r') as index_add_dict_jsn:
            self.index_add_dict = json.load(index_add_dict_jsn)

        if not os.path.exists(os.path.join(self.config.temp_path, "index_exchange_line_dict.json")):
            self.obfuscation.create_exchange_dict()
        with open(os.path.join(self.config.temp_path, "index_exchange_line_dict.json"), 'r') as index_exchange_line_dict_jsn:
            self.index_exchange_line_dict = json.load(index_exchange_line_dict_jsn)

        if not os.path.exists(os.path.join(self.config.temp_path, "index_merge_dict.json")):
            self.obfuscation.create_merge_line_dict()
        with open(os.path.join(self.config.temp_path, "index_merge_dict.json"), 'r') as index_merge_dict_jsn:
            self.index_merge_dict = json.load(index_merge_dict_jsn)

        if not os.path.exists(os.path.join(self.config.temp_path, "index_replace_dict.json")):
            self.obfuscation.create_replace_dict()
        with open(os.path.join(self.config.temp_path, "index_replace_dict.json"), 'r') as index_replace_dict_jsn:
            self.index_replace_dict = json.load(index_replace_dict_jsn)

        with open(os.path.join(self.config.temp_path, "index_line_mutation_dict.json"), 'r') as index_line_mutation_dict_jsn:
            self.index_line_mutation_dict = json.load(index_line_mutation_dict_jsn)

        with open(os.path.join(self.config.temp_path, "index_label.json"), 'r') as index_label_jsn:
            self.index_label_dict = json.load(index_label_jsn)

    def exchange_line(self, sample_slice, line):
        sample_slice[line], sample_slice[line+1] = sample_slice[line+1], sample_slice[line]
        return sample_slice

    def get_best_adv(self, adv_slice_list, ori_label, ori_prob):
        adv_slice_list_trans = []
        for adv in adv_slice_list:
            adv_slice_list_trans.append(self.list_to_statement(adv))
        dataloader = self.obfuscation.make_batch(adv_slice_list_trans, norm=True)
        self.check_model_num += len(dataloader)
        preds, probs = self.obfuscation.run_model_by_batch(dataloader)
        adv_score_list = []
        i = 0
        while i < len(preds):
            label = preds[i]
            score = probs[i]
            if label != ori_label:
                state_score = ori_prob - (1-score)
            else:
                state_score = ori_prob - score
            adv_score_list.append(state_score)
            i += 1
        state_score_arr = np.array(adv_score_list)
        sort_index = np.argsort(-state_score_arr)
        return adv_slice_list[sort_index[0]]

    def add_dead_code(self, sample_slice, line, variable_declaration_list, ori_label, ori_prob, best_word, method = 1):
        if method == 0:
            best_words = self.get_candidates_by_frequency(sample_slice, ori_label)
        elif method == 1:
            best_words = self.get_candidates_by_token_importance(sample_slice, ori_label)
        elif method == 2:
            best_words = [["unknown_" + str(j)] for j in range(self.replace_limit_number)]
        elif method == 3:
            best_words = self.nope_tool.get_candidate_stament(ori_label)
        else:
            best_words = [best_word]
        adv_slice_list = []
        for variable_declaration in variable_declaration_list:
            for b in best_words:
                best_word = b[0]
                if ori_label == 0:
                    insert_statement_dead_code = self.obfuscation.insert_statement0(variable_declaration, best_word)
                else:
                    insert_statement_dead_code = self.obfuscation.insert_statement1(variable_declaration, best_word)
                insert_statement_dead_code = sample_slice[line-1] + insert_statement_dead_code
                adv_slice = sample_slice[:line-1] + [insert_statement_dead_code] + sample_slice[line:]
                adv_slice_list.append(adv_slice)
        return self.get_best_adv(adv_slice_list, ori_label, ori_prob)

    def const_replace(self, sample_slice, line, replace_content):
        adv_slice = sample_slice[0:line] + [replace_content] + sample_slice[line+1:]
        return adv_slice

    def mutation_repalce(self, sample_slice, line, mutation_contents, ori_label, ori_prob):
        adv_slice_list = []
        for mutation_line in mutation_contents:
            adv_slice = sample_slice[:line] + [[mutation_line]] + sample_slice[line+1:]
            adv_slice_list.append(adv_slice)
        return self.get_best_adv(adv_slice_list, ori_label, ori_prob)

    def merge_func(self, sample_slice, start_line, end_line):
        if start_line == end_line:
            insert = sample_slice[start_line][1:]
        else:
            insert = sample_slice[start_line][1:] + sample_slice[end_line][1:]
        adv_slice = sample_slice[:start_line] + [insert] + sample_slice[end_line+1:]
        return adv_slice

    def get_candidates_by_frequency(self, sample_slice, ori_label):
        sample_slice = self.list_to_statement(sample_slice)
        if not os.path.exists(os.path.join(self.config.temp_path, "token_frequency.json")):
            test_data = pd.read_pickle(self.config.data_path + 'test/blocks.pkl')
            token_frequency_dict = {0:{}, 1:{}}
            for index, program in test_data.iterrows():
                slice = program['orig_code']
                label = program['label']
                for sentence in slice:
                    tokens = create_tokens(sentence)
                    for token in tokens:
                        if token in token_frequency_dict[label].keys():
                            token_frequency_dict[label][token] += 1
                        else:
                            token_frequency_dict[label][token] = 1
            with open(os.path.join(self.config.temp_path, "token_frequency.json"), 'w') as token_frequency_jsn:
                json.dump(token_frequency_dict, token_frequency_jsn)
        if self.token_list == None:
            with open(os.path.join(self.config.temp_path, "token_frequency.json"), 'r') as token_frequency_jsn:
                token_frequency_dict = json.load(token_frequency_jsn)
            self.token_list = token_frequency_dict
        token_frequency_dict = self.token_list
        curr_tokens = []
        for sentence in sample_slice:
            curr_tokens.extend(create_tokens(sentence))
        tmp = token_frequency_dict[str(1-ori_label)]
        tmp_list = []
        for token in tmp.keys():
            tmp_list.append([token, int(tmp[token])])
        tmp_list.sort(key = lambda  x : x[1], reverse=True)
        ret = []
        count = 3
        for t in tmp_list:
            if t[0] in curr_tokens:
                continue
            ret.append([t[0]])
            count -= 1
            if count <= 0:
                break
        return ret

    def token_impact(self, ori_label, ori_prob, s_lengths, token_symbol_sequence):
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
        dataloader = self.obfuscation.make_batch(map_programs, norm=True)
        preds, probs = self.nope_tool.run_model_by_batch(dataloader, ori_label)

        state_score_arr = np.array(probs)
        sort_index = np.argsort(state_score_arr)
        p = []
        for i in sort_index:
            p.append(probs[i])
        return sort_index, p

    def get_candidates_by_token_importance(self, sample_slice, ori_label):
        sample_slice = self.list_to_statement(sample_slice)
        if not os.path.exists(os.path.join(self.config.temp_path, "token_importance_" +self.model_name+ ".json")):
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
                map = self.obfuscation.norm_program(slice)
                ori_label_new, ori_prob = self.obfuscation.predict_adv_program(map)
                token_symbol_sequence = []
                [token_symbol_sequence.extend(create_tokens(statement)) for statement in slice]
                _, s_lengths = DataPipline.states2idseq(slice, self.config.vocab, self.config.vocab_size - 1)
                sort_index, probs = self.token_impact(ori_label_new, ori_prob, s_lengths, token_symbol_sequence)
                for pos in range(2):
                    i = sort_index[pos]
                    prob = probs[pos]
                    if token_symbol_sequence[i] in token_importance_dict[ori_label_new].keys():
                        # 按变化概率确定词的评分
                        token_importance_dict[ori_label_new][token_symbol_sequence[i]] = max(token_importance_dict[ori_label_new][token_symbol_sequence[i]], ori_prob-prob)
                    else:
                        token_importance_dict[ori_label_new][token_symbol_sequence[i]] = ori_prob - prob
            with open(os.path.join(self.config.temp_path, "token_importance_" + self.model_name+ ".json"), 'w') as token_importance_jsn:
                json.dump(token_importance_dict, token_importance_jsn)
        if self.token_list == None:
            with open(os.path.join(self.config.temp_path, "token_importance_" + self.model_name+ ".json"), 'r') as token_importance_jsn:
                token_importance_dict = json.load(token_importance_jsn)
            self.token_list = token_importance_dict
        token_importance_dict = self.token_list
        curr_tokens = []
        for sentence in sample_slice:
            curr_tokens.extend(create_tokens(sentence))
        tmp = token_importance_dict[str(1-ori_label)]
        tmp_list = []
        for token in tmp.keys():
            tmp_list.append([token, int(tmp[token])])
        tmp_list.sort(key = lambda x : x[1], reverse=True)
        ret = []
        count = 5
        for t in tmp_list:
            if t[0] in curr_tokens or t[0].upper() in curr_tokens:
                continue
            ret.append([t[0]])
            count -= 1
            if count <= 0:
                break
        return ret

    def macro_replace(self, sample_slice, line, ori_label, ori_prob, macro_replaced_limit=1, method = 1):
        if method == 0:
            candidates = self.get_candidates_by_frequency(sample_slice, ori_label)
        elif method == 1:
            candidates = self.get_candidates_by_token_importance(sample_slice, ori_label)
        else:
            candidates = [["unknown_" + str(j)] for j in range(self.replace_limit_number)]
        # 在当前行中替换
        if line>=len(sample_slice) or len(sample_slice[line])<1:
            return sample_slice
        # print(line, len(sample_slice))
        s_split = create_tokens(sample_slice[line][-1].lower())
        if s_split == []:
            s_split = ['\n']
        res = sample_slice[line][:-1]
        tmp_list = []
        for i, _ in enumerate(s_split):
            tmp_line = s_split[:i] + s_split[i+1:]
            tmp_slice = sample_slice[0:line] + [res + [" ".join(tmp_line)]] + sample_slice[line+1:]
            tmp_list.append(tmp_slice)
        tmp_list_trans = []
        for x in tmp_list:
            tmp_list_trans.append(self.list_to_statement(x))
        dataloader = self.obfuscation.make_batch(tmp_list_trans, norm=True)
        self.check_model_num += len(dataloader)
        preds, probs = self.obfuscation.run_model_by_batch(dataloader)
        adv_score_list = []
        i = 0
        while i < len(preds):
            label = preds[i]
            score = probs[i]
            if label != ori_label:
                state_score = ori_prob - (1-score)
            else:
                state_score = ori_prob - score
            adv_score_list.append(state_score)
            i += 1
        state_score_arr = np.array(adv_score_list)
        sort_index = np.argsort(-state_score_arr)
        # r = min(self.replace_limit_number, len(sort_index))
        r = macro_replaced_limit
        token_location = sort_index[:r]
        adv_slice_list = []
        s_split_list = [s_split]
        candidates = candidates[:r]
        candidates_list = [candidates]
        for l in token_location:
            tmp = []
            for i,s_split in enumerate(s_split_list):
                candidates = candidates_list[i]
                for candidate in candidates:
                    macro_replace_line = s_split[:l] + [candidate[0].upper()] + s_split[l+1:]
                    # macro_replace_line = s_split[:l] + candidate+ s_split[l + 1:]
                    # if macro_replace_line not in tmp:
                    tmp.append(macro_replace_line)
                    t_c = copy.deepcopy(candidates)
                    t_c.remove(candidate)
                    candidates_list.append(t_c)
                    # macro_replace_line = ["unknown_0"] # 效果不好
                    adv_slice = sample_slice[0:line] + [res + [" ".join(macro_replace_line)]] + sample_slice[line+1:]
                    adv_slice_list.append(adv_slice)
            # candidates.remove(candidates[0])
            s_split_list.extend(tmp)

        # 在全部的词中替换
        # token_sequence, s_lengths = DataPipline.states2idseq(sample_slice, self.config.vocab, self.config.vocab_size - 1)
        # token_symbol_sequence = []
        # [token_symbol_sequence.extend(create_tokens(statement)) for statement in sample_slice]
        # sort_index = self.obfuscation.token_impact_blackbox(ori_label, ori_prob, s_lengths, token_symbol_sequence)
        # token_location = sort_index[:self.replace_limit_number]
        # adv_slice_list = [sample_slice]
        # for l in token_location:
        #     statement, line_num, token_line_index = self.obfuscation.token_in_line(l, token_symbol_sequence, s_lengths)
        #     tmp = []
        #     for s in adv_slice_list:
        #         s_split = create_tokens(s[line_num])
        #         for candidate in candidates:
        #             replace_line = s_split[:token_line_index] + candidate + s_split[token_line_index+1:]
        #             adv_slice = s[:line_num] + [" ".join(replace_line)] + s[line_num+1:]
        #             if adv_slice not in tmp:
        #                 tmp.append(adv_slice)
        #     adv_slice_list.extend(tmp)
        return self.get_best_adv(adv_slice_list, ori_label, ori_prob)

    def statement_to_list(self, sample_slice):
        # 把[" ", " ", ...] 转换成 [[" "], [" "]...]的形式 方便将改动限制在一行内 而不用改变行号
        l_sample_slice = []
        for statement in sample_slice:
            l_sample_slice.append([statement])
        return l_sample_slice

    def list_to_statement(self, l_sample_slice):
        # 转回来
        sample_slice = []
        for l in l_sample_slice:
            if l == []:
                continue
            for statement in l:
                sample_slice.append(statement)
        return sample_slice

    def combination_attack(self, lines_limit=2, random_tag=False):
        count = 0
        sucess = 0
        if random_tag:
            method = 2
            m_name = "random"
        else:
            method = 1
            m_name = "greedy"
        failed = []
        # if self.replace_limit_number>7:
        #     macro_replaced_limit = 3
        # else:
        #     macro_replaced_limit = 1
        macro_replaced_limit = 3
        mutate_tag, exchange_tag, add_tag, merge_tag, const_tag, macro_tag, unroll_loop = True, True, True, True, True, True, True
        # mutate_tag, exchange_tag, add_tag, merge_tag, const_tag, macro_tag, unroll_loop = True, True, True, True, True, True, True
        with open(os.path.join(self.config.result_path, "combination_" + self.model_name +"(00000,"+m_name+","+str(self.replace_limit_number) + ").csv"), 'w', encoding='utf-8', newline='') as r_csv_f:
            csv_writer = csv.writer(r_csv_f)
            csv_writer.writerow(['index', 'true_label', 'ori_label', 'ori_prob', 'ori_code', 'adv_label', 'adv_prob', 'adv_code', 'query_times'])
            for index in self.sample_ids:
                index = str(index)
                self.check_model_num = 0
                is_sucess = False
                diff = 0
                ground_label = int(self.index_label_dict[index])
                add_dict = self.index_add_dict[index]
                exchange_line_dict = self.index_exchange_line_dict[index]
                replace_dict = self.index_replace_dict[index]
                merge_dict = self.index_merge_dict[index]
                mutation_dict = self.index_line_mutation_dict[index]
                with open(os.path.join(self.config.sample_slice_path, index), 'r') as sample_slice_f:
                    sample_slice_ori = sample_slice_f.readlines()
                sample_slice = self.statement_to_list(sample_slice_ori)
                # with open(os.path.join(self.config.sample_path, index), 'r') as map_program_f:
                #     map_program = map_program_f.readlines()
                map_program = self.obfuscation.norm_program(sample_slice_ori)
                ori_label, ori_prob = self.obfuscation.predict_adv_program(map_program)
                save_ori_prob = ori_prob
                self.check_model_num += 1
                if ori_label != ground_label:
                    sample_slice_ori = concat_statement(sample_slice_ori)
                    csv_writer.writerow([index, ground_label, ori_label, ori_prob, sample_slice_ori, ori_label, ori_prob, sample_slice_ori, self.check_model_num])
                    continue
                count += 1
                # if count <= 550:
                #     continue
                if not random_tag:
                    sorted_lines = self.obfuscation.statement_impact(map_program, ori_label, ori_prob)
                else:
                    lines_number = [i for i in range(len(map_program))]
                    np.random.seed(int(time.time()))
                    # sorted_lines = np.random.choice(lines_number, size=lines_limit)
                    sorted_lines = np.random.choice(lines_number, size=len(map_program))
                # modified_lines = min(len(sample_slice), lines_limit)
                for important_line in sorted_lines:
                    if diff >= self.replace_limit_number:
                        break
                    best_word = map_program[sorted_lines[-1]]
                    if not is_sucess and str(important_line) in exchange_line_dict.keys() and diff < self.replace_limit_number and exchange_tag:
                        # 先做行交换
                        adv = self.exchange_line(sample_slice, important_line)
                        adv_tans = self.list_to_statement(adv)
                        adv_map = self.obfuscation.norm_program(adv_tans)
                        adv_label, adv_prob = self.obfuscation.predict_adv_program(adv_map)
                        self.check_model_num += 1
                        if adv_label != ground_label:
                            sample_slice = adv
                            is_sucess = True
                            diff += 1
                        else:
                            if adv_prob < ori_prob:
                                ori_prob = adv_prob
                                sample_slice = adv
                                important_line += 1
                                diff += 1
                    if not is_sucess and str(important_line-1) in add_dict.keys() and diff < self.replace_limit_number and add_tag:
                        # 插入死代码
                        variable_declaration_list = []
                        if important_line > 0:
                            variable_declaration_list = add_dict[str(important_line-1)]
                        if variable_declaration_list != []:
                            adv = self.add_dead_code(sample_slice, important_line, variable_declaration_list, ori_label, ori_prob, best_word, method = method)
                            adv_tans = self.list_to_statement(adv)
                            adv_map = self.obfuscation.norm_program(adv_tans)
                            adv_label, adv_prob = self.obfuscation.predict_adv_program(adv_map)
                            self.check_model_num += 1
                            if adv_label != ground_label:
                                sample_slice = adv
                                is_sucess = True
                                diff += 1
                            else:
                                if adv_prob < ori_prob:
                                    ori_prob = adv_prob
                                    sample_slice = adv
                                    # important_line += 3
                                diff += 1
                    if not is_sucess and str(important_line-1) in replace_dict.keys() and diff < self.replace_limit_number and const_tag:
                        # 插入常量替换
                        replace_content = replace_dict[str(important_line-1)] # 生成的字典行号差1
                        adv = self.const_replace(sample_slice, important_line, replace_content)
                        adv_tans = self.list_to_statement(adv)
                        adv_map = self.obfuscation.norm_program(adv_tans)
                        adv_label, adv_prob = self.obfuscation.predict_adv_program(adv_map)
                        self.check_model_num += 1
                        if adv_label != ground_label:
                            sample_slice = adv
                            is_sucess = True
                            diff += 1
                        else:
                            if adv_prob < ori_prob:
                                ori_prob = adv_prob
                                sample_slice = adv
                                # important_line += len(replace_content) - 1
                                diff += 1
                    if not is_sucess and str(important_line) in mutation_dict.keys() and diff < self.replace_limit_number and mutate_tag:
                        # 变异行替换
                        mutation_contents = mutation_dict[str(important_line)]
                        if mutation_contents != []:
                            adv = self.mutation_repalce(sample_slice, important_line, mutation_contents, ori_label, ori_prob)
                            adv_tans = self.list_to_statement(adv)
                            adv_map = self.obfuscation.norm_program(adv_tans)
                            adv_label, adv_prob = self.obfuscation.predict_adv_program(adv_map)
                            self.check_model_num += 1
                            if adv_label != ground_label:
                                sample_slice = adv
                                is_sucess = True
                                diff += 1
                            else:
                                if adv_prob < ori_prob:
                                    ori_prob = adv_prob
                                    sample_slice = adv
                                    diff += 1
                    if not is_sucess and str(important_line) in merge_dict.keys() and diff < self.replace_limit_number and merge_tag:
                        # 合并函数
                        start_line, end_line = important_line, int(merge_dict[str(important_line)])
                        adv = self.merge_func(sample_slice, start_line, end_line)
                        adv_tans = self.list_to_statement(adv)
                        adv_map = self.obfuscation.norm_program(adv_tans)
                        adv_label, adv_prob = self.obfuscation.predict_adv_program(adv_map)
                        self.check_model_num += 1
                        if adv_label != ground_label:
                            sample_slice = adv
                            is_sucess = True
                            diff += 1
                        else:
                            if adv_prob < ori_prob:
                                ori_prob = adv_prob
                                sample_slice = adv
                                diff += 1
                    if macro_tag and not is_sucess and diff < self.replace_limit_number:
                        # 宏替换
                        adv = self.macro_replace(sample_slice, important_line, ori_label, ori_prob, macro_replaced_limit, method = method)
                        adv_tans = self.list_to_statement(adv)
                        adv_map = self.obfuscation.norm_program(adv_tans)
                        adv_label, adv_prob = self.obfuscation.predict_adv_program(adv_map)
                        self.check_model_num += 1
                        if adv_label != ground_label:
                            sample_slice = adv
                            is_sucess = True
                            diff += macro_replaced_limit
                        else:
                            if adv_prob < ori_prob:
                                ori_prob = adv_prob
                                sample_slice = adv
                                diff += macro_replaced_limit
                    if unroll_loop and not is_sucess and diff < self.replace_limit_number:
                        adv = self.unroll_loop(sample_slice, important_line)
                        adv_tans = self.list_to_statement(adv)
                        adv_map = self.obfuscation.norm_program(adv_tans)
                        adv_label, adv_prob = self.obfuscation.predict_adv_program(adv_map)
                        self.check_model_num += 1
                        if adv_label != ground_label:
                            sample_slice = adv
                            is_sucess = True
                            diff += 1
                        else:
                            if adv_prob < ori_prob:
                                ori_prob = adv_prob
                                sample_slice = adv
                            diff += 1
                if is_sucess:
                    sucess += 1
                else:
                    failed.append(index)
                # sample_slice一直保存最优的对抗样本
                adv_tans = self.list_to_statement(sample_slice)
                adv_map = self.obfuscation.norm_program(adv_tans)
                adv_label, adv_prob = self.obfuscation.predict_adv_program(adv_map)
                sample_slice_ori = concat_statement(sample_slice_ori)
                adv_tans = concat_statement(adv_tans)
                csv_writer.writerow([index, ground_label, ori_label, save_ori_prob, sample_slice_ori, adv_label, adv_prob, adv_tans, self.check_model_num])
                print(sucess, count, sucess / count)
            csv_writer.writerow([sucess, count, sucess / count])
            csv_writer.writerow(failed)
            # print(self.nope_tool.ret)

    def unroll_loop(self, sample_slice, important_line):
        flag = ''.join(random.choice(string.ascii_uppercase) for i in range(5))
        if sample_slice[important_line] == []:
            return sample_slice
        line = sample_slice[important_line][-1]
        res = sample_slice[important_line][:-1]
        line_tokens = create_tokens(line)
        if "while" in line_tokens:
            r1 = re.compile(r'[(](.*)[)]', re.S)
            exp = re.findall(r1, line)
            if exp == []:
                return sample_slice
            statements = ["bool " + flag + " =true\n", "while(" + flag + "){\n",]
            statements1 = ["if(!(" + exp[0] + "))" + flag + "=false;\n"]
            new_line = res + statements
            new_line1 = statements1 + sample_slice[important_line+1]
            return sample_slice[:important_line] + [new_line] + [new_line1] + sample_slice[important_line+2:]
        elif "for" in line_tokens:
            r1 = re.compile(r'[(](.*)[)]', re.S)
            exp = re.findall(r1, line)
            i_s = exp[0].split(';')
            if len(i_s) != 3:
                return sample_slice
            statements = ["bool " + flag + " =true\n", "for(" + i_s[0] + ';' + flag + ';' + i_s[2] + ")\n"]
            statements1 = ["if(!(" + i_s[1] + "))" + flag + "=false;\n"]
            new_line = res + statements
            new_line1 = statements1 + sample_slice[important_line + 1]
            return sample_slice[:important_line] + [new_line] + [new_line1] + sample_slice[important_line + 2:]
        else:
            return sample_slice

    def only_one_sample(self):
        # 个例查看攻击情况
        index = '197366'
        with open(os.path.join(self.config.sample_slice_path, index), 'r') as f:
            slice = f.readlines()
        # slice = ['\n']
        ground_label = int(self.index_label_dict[index])
        map = self.obfuscation.norm_program(slice)
        ori_label, ori_prob = self.obfuscation.predict_adv_program(map)
        sorted_lines = self.obfuscation.statement_impact(map, ori_label, ori_prob)
        with open(os.path.join(self.config.sample_slice_path, '175407'), 'r') as f1:
            vul = f1.readlines()
        vul_text = ''
        for s in vul:
            vul_text += s + ' '
        v = self.index_add_dict[index][str(len(slice)-1)][0]
        statement_insert = self.obfuscation.insert_statement0(v, vul_text)
        slice1 = slice[1:]
        l1, p1 = self.obfuscation.predict_adv_program(self.obfuscation.norm_program(slice1))
        adv_slice = slice[:sorted_lines[0]] + slice[sorted_lines[0]+1:]
        adv_slice = adv_slice[10:]
        map2 = self.obfuscation.norm_program(adv_slice)
        l2, p2 = self.obfuscation.predict_adv_program(map2)
        with open(os.path.join(self.config.sample_path, index), 'r') as f1:
            map = f1.readlines()
        l3, p3 = self.obfuscation.predict_adv_program(map)
        adv_list = []
        for l in range(len(slice)):
            adv_list.append(self.obfuscation.norm_program(slice[:l]+slice[l+1:]))
        dataloader = self.obfuscation.make_batch(adv_list)
        preds, probs = self.nope_tool.run_model_by_batch(dataloader, ori_label)
        state_score_arr = np.array(probs)
        sort_index = np.argsort(state_score_arr)
        adv1 = self.macro_replace(slice, sorted_lines[0], ori_label, ori_prob, method=1)
        if str(sort_index[0]) in self.index_add_dict[index].keys():
            v_l = self.index_add_dict[index][str(sort_index[0])]
        adv2 = self.add_dead_code(slice, sort_index[0], self.index_add_dict[index][str(sort_index[0])], ori_label, ori_prob, ' ', method=1)
        la1,pro1 = self.obfuscation.predict_adv_program(self.obfuscation.norm_program(adv1))
        la2,pro2 = self.obfuscation.predict_adv_program(self.obfuscation.norm_program(adv2))

        print(ori_label, ori_prob, l1, p1, l2, p2, l3, p3)
        print(slice[sorted_lines[0]])

