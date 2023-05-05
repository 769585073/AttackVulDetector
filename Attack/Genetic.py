import copy
import json
import os
import csv
import torch
import random
import time
import string
import re

import torch.nn.functional as F
import numpy as np

from DataProcess.DataPipline import DataPipline
from Utils.get_tokens import create_tokens
from Utils.mapping import mapping
from Target_model.RNNDetect import DetectModel
from Utils.Util import concat_statement


class Population():
    def __init__(self, config, sample_slice, index, ori_label):
        self.config = config
        self.index = index
        self.ori_label = ori_label
        self.sample_slice = sample_slice
        self.translit_slice = None
        self.members = []
        self.translit_members = []
        self.used_method = []
        self.line_replaced = {}
        self.fitness_scores = None
        self.macro_names = None
        self.result = sample_slice
        self.used_method_result = None
        self.translit_result = None


class GeneticAlgorithm():
    def __init__(self, config, limits, model_name = 'VulDetectModel.pt', pop_size=60, max_iters=20):
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.check_model_num = 0
        self.config = config
        self.model_name = model_name
        self.limits = limits
        self.ratio = {"select":0.4, "cross":0.6, "mutate":0.6}
        # self.ratio = {"select": 1.0, "cross": 0.0, "mutate": 0.8}
        # self.ratio = {"select": 0, "cross": 0, "mutate": 1}
        self.load_temp_files()
        # 专门用于只跑变异的情况（贪心选行）
        self.sorted_index = []

    def load_temp_files(self):
        detect_model = DetectModel(self.config)
        if self.config.use_gpu:
            detect_model.cuda()
        detect_model.load_state_dict(torch.load(self.config.models_path + self.model_name))
        self.detect_model = detect_model

        with open(os.path.join(self.config.temp_path, "index_add_dict.json"), 'r') as index_add_dict_jsn:
            self.index_add_dict = json.load(index_add_dict_jsn)

        with open(os.path.join(self.config.temp_path, "index_exchange_line_dict.json"), 'r') as index_exchange_line_dict_jsn:
            self.index_exchange_line_dict = json.load(index_exchange_line_dict_jsn)

        with open(os.path.join(self.config.temp_path, "index_merge_dict.json"), 'r') as index_merge_dict_jsn:
            self.index_merge_dict = json.load(index_merge_dict_jsn)

        with open(os.path.join(self.config.temp_path, "index_replace_dict.json"), 'r') as index_replace_dict_jsn:
            self.index_replace_dict = json.load(index_replace_dict_jsn)

        with open(os.path.join(self.config.temp_path, "index_line_mutation_dict.json"), 'r') as index_line_mutation_dict_jsn:
            self.index_line_mutation_dict = json.load(index_line_mutation_dict_jsn)

        with open(os.path.join(self.config.temp_path, "token_importance_" + self.model_name+ ".json"), 'r') as token_importance_jsn:
            token_importance = json.load(token_importance_jsn)
        self.label_important_tokens = {}
        for label in token_importance.keys():
            tmp = []
            for token in token_importance[label]:
                tmp.append([token, int(token_importance[label][token])])
            tmp.sort(key = lambda  x : x[1], reverse=True)
            self.label_important_tokens[int(label)] = tmp

    def create_candidates(self, population):
        mutate_tag, exchange_tag, add_tag, merge_tag, const_tag, macro_tag, unroll_loop = True, True, True, True, True, True, True
        n = len(population.sample_slice)
        curr_tokens = []
        for sentence in population.sample_slice:
            curr_tokens.extend(create_tokens(sentence))
        macro_names = []
        for token in self.label_important_tokens[population.ori_label]:
            if token[0] in curr_tokens:
                continue
            macro_names.append(token[0].upper())
        population.macro_names = macro_names
        for line in range(n):
            tmp = []
            if str(line) in self.index_line_mutation_dict[population.index].keys() and mutate_tag:
                tmp.append(["mutate", self.index_line_mutation_dict[population.index][str(line)]])
            if str(line) in self.index_exchange_line_dict[population.index].keys() and exchange_tag:
                tmp.append(["exchange", self.index_exchange_line_dict[population.index][str(line)]])
            if str(line-1) in self.index_add_dict[population.index].keys() and self.index_add_dict[population.index][str(line-1)]!=[] and add_tag:
                tmp.append(["add", self.index_add_dict[population.index][str(line-1)]])
            if str(line) in self.index_merge_dict[population.index].keys() and merge_tag:
                tmp.append(["merge", self.index_merge_dict[population.index][str(line)]])
            if str(line-1) in self.index_replace_dict[population.index].keys() and const_tag: # const的字典行号比真实行号小1
                tmp.append(["const", self.index_replace_dict[population.index][str(line-1)]])
            if macro_tag:
                tmp.append(["macro", None])
            if unroll_loop:
                tmp.append(["loop", None])
            population.line_replaced[line] = tmp

    def statement_impact(self, program_slice, ori_label):
        map_programs = []
        for state_index in range(len(program_slice)):
            _temp_map_program = program_slice[:state_index] + program_slice[state_index + 1:]
            map_programs.append(_temp_map_program)
        dataloader = self.make_batch(map_programs, norm=True)
        preds, probs = self.fitness(dataloader, ori_label)
        state_score_arr = np.array(probs)
        sort_index = np.argsort(state_score_arr)
        return sort_index

    def insert_statement1(self, variable_declaration, best_word):
        line1 = 'printf(\"' + best_word + ', %p\n' + '\", &' + variable_declaration + ');'
        return [line1]

    def insert_statement0(self, variable_declaration, best_word):
        # line1 = 'if(' + variable_declaration + '!=' + variable_declaration + '){(char *) ' + variable_declaration + '=\"' + best_word + '\";}'
        line1 = 'printf(\"' + best_word + ', %p\n' + '\", &' + variable_declaration + ');'
        return [line1]

    def create_population_member(self, translit_slice, used_method, candidate_info, line, ori_label, candidate_words, used_macro_all):
        # 按照方法生成种群成员 used_macro_all该样本使用过的宏名称
        if candidate_info[0] == "mutate":
            new_line = random.choice(candidate_info[1])
            if used_method["mutate"] == 0:
                used_method["mutate"] = 1
                if used_method["merge"] == 0:
                    if translit_slice[line] != []:
                        translit_slice[line][0] = new_line
                else:
                    translit_slice[line] = [new_line] + translit_slice[line]
        elif candidate_info[0] == "exchange":
            exchange_line = int(candidate_info[1])
            if exchange_line < len(translit_slice):
                translit_slice[line], translit_slice[exchange_line] = translit_slice[exchange_line], translit_slice[line]
                used_method["exchange"] = exchange_line
        elif candidate_info[0] == "add":
            variable_declaration = random.choice(candidate_info[1])
            word = random.choice(candidate_words)
            if ori_label == 0:
                insert_statement_dead_code = self.insert_statement0(variable_declaration, word)
            else:
                insert_statement_dead_code = self.insert_statement1(variable_declaration, word)
            insert_statement_dead_code = translit_slice[line-1] + insert_statement_dead_code
            translit_slice = translit_slice[:line-1] + [insert_statement_dead_code] + translit_slice[line:]
            used_method["add"] += 1
        elif candidate_info[0] == "merge":
            if used_method["merge"] == 0:
                if line == int(candidate_info[1]):
                    new_line = translit_slice[line][1:]
                else:
                    new_line = translit_slice[line][1:] + translit_slice[int(candidate_info[1])][1:]
                translit_slice = translit_slice[:line] + [new_line] + translit_slice[int(candidate_info[1])+1:]
                used_method["merge"] = 1
        elif candidate_info[0] == "const":
            if used_method["const"] == 0:
                const_declaration = translit_slice[line-1] + candidate_info[1][:-1]
                new_line = candidate_info[1][-1:] + translit_slice[line][1:]
                translit_slice = translit_slice[:line-1] + [const_declaration] + [new_line] + translit_slice[line+1:]
                used_method["const"] = 1
        elif candidate_info[0] == "loop":
            if used_method["loop"] == 0 and translit_slice[line]!=[] and line<len(translit_slice)-1:
                flag = ''.join(random.choice(string.ascii_uppercase) for i in range(5))
                line_content = translit_slice[line][-1]
                res = translit_slice[line][:-1]
                line_tokens = create_tokens(line_content)
                if "while" in line_tokens:
                    r1 = re.compile(r'[(](.*)[)]', re.S)
                    exp = re.findall(r1, line_content)
                    if exp != []:
                        statements = ["bool " + flag + " =true\n", "while(" + flag + "){\n", ]
                        statements1 = ["if(!(" + exp[0] + "))" + flag + "=false;\n"]
                        new_line = res + statements
                        new_line1 = statements1 + translit_slice[line + 1]
                        translit_slice = translit_slice[:line] + [new_line] + [new_line1] + translit_slice[line + 2:]
                        used_method["loop"] = 1
                elif "for" in line_tokens:
                    r1 = re.compile(r'[(](.*)[)]', re.S)
                    exp = re.findall(r1, line_content)
                    if exp:
                        i_s = exp[0].split(';')
                        if len(i_s) == 3:
                            statements = ["bool " + flag + " =true\n", "for(" + i_s[0] + ';' + flag + ';' + i_s[2] + ")\n"]
                            statements1 = ["if(!(" + i_s[1] + "))" + flag + "=false;\n"]
                            new_line = res + statements
                            new_line1 = statements1 + translit_slice[line + 1]
                            translit_slice = translit_slice[:line] + [new_line] + [new_line1] + translit_slice[line + 2:]
                            used_method["loop"] = 1
        else:
            candidate_words = set(candidate_words) - used_macro_all
            if candidate_words!=set() and line<len(translit_slice) and translit_slice[line]!=[]:
                word = random.choice(list(candidate_words))
                # word = ''.join(random.choice(string.ascii_uppercase) for i in range(5))
                used_method["macro"].add(word)
                curr_tokens = create_tokens(translit_slice[line][0].lower())
                if len(curr_tokens) > 0:
                    replace_location = random.randint(0, len(curr_tokens)-1)
                    new_tokens = curr_tokens[:replace_location] + [word] + curr_tokens[replace_location+1:]
                    new_line = [" ".join(new_tokens)] + translit_slice[line][1:]
                    translit_slice = translit_slice[:line] + [new_line] + translit_slice[line+1:]
        return translit_slice, used_method

    def check_limits(self, population):
        # statement_limit = self.limits[0]
        # macro_limit = self.limits[1]
        members = []
        translit_members = []
        used_method = []
        for pos, member_used_method in enumerate(population.used_method):
            # curr_statement_modify, curr_macro_modify = 0, 0
            # for line_used_method in member_used_method[:-1]:
            #     line_modify = 0
            #     for method, count in line_used_method.items():
            #         if method == "macro":
            #             curr_macro_modify += len(count)
            #         else:
            #             if count > 0:
            #                 line_modify = 1
            #     if line_modify==1:
            #         curr_statement_modify += 1
            # if curr_statement_modify > statement_limit or curr_macro_modify > macro_limit:
            #     continue
            curr_modify = 0
            for line_used_method in member_used_method[:-1]:
                for method, count in line_used_method.items():
                    if method == "macro":
                        curr_modify += len(count)
                    elif method == "exchange" and count>0:
                        curr_modify += 0.5
                    elif method == "exchange":
                        curr_modify += 0
                    else:
                        curr_modify += count
            if curr_modify<=self.limits and len(population.translit_members[pos]) == len(population.translit_slice):
                members.append(population.members[pos])
                translit_members.append(population.translit_members[pos])
                used_method.append(population.used_method[pos])
        if len(members) == 0:
            for _ in range(self.pop_size):
                members.append(copy.deepcopy(population.result))
                translit_members.append(copy.deepcopy(population.translit_slice))
                used_method.append(copy.deepcopy(population.used_method_result))
        else:
            for _ in range(self.pop_size-len(members)):
                copy_pos = random.randint(0, len(members)-1)
                members.append(copy.deepcopy(members[copy_pos]))
                translit_members.append(copy.deepcopy(translit_members[copy_pos]))
                used_method.append(copy.deepcopy(used_method[copy_pos]))
        population.members = members
        population.translit_members = translit_members
        population.used_method = used_method

    def norm_program(self, program_slice):
        if program_slice == None:
            return []
        inst_statements = []
        for line in program_slice:
            token_list = create_tokens(line)
            inst_statements.append(token_list)
        map_program, _ = mapping(inst_statements)
        return map_program

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
            input_seq = DataPipline.states2idseq(map_program, self.config.vocab, self.config.vocab_size-1)
            batch.append(input_seq)
        if batch != []:
            dataset.append(batch)
        return dataset

    def predict_single(self, program_slice, norm = True):
        if norm:
            map_slice = self.norm_program(program_slice)
        else:
            map_slice = program_slice
        self.check_model_num += 1
        input = DataPipline.states2idseq(map_slice, self.config.vocab, self.config.vocab_size-1)
        output = self.detect_model([input])
        predicted = torch.max(output, 1)[1].detach().cpu().data.numpy().tolist()[0]
        probability = F.softmax(output,dim=1).detach().cpu().data.numpy().tolist()[0][predicted]
        return predicted, probability

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

    def fitness(self, dataloader, ori_label):
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
        self.check_model_num += len(dataloader)
        return predicted, probability

    def initial_population(self, population):
        used_method_init = {"mutate":0, "exchange":-1, "add":0, "merge":0, "const":0, "macro":set(), "loop":0}
        self.create_candidates(population)
        population.translit_slice = self.statement_to_list(population.sample_slice)
        n = len(population.sample_slice)
        member_used_method = []
        for l in range(n):
            member_used_method.append(copy.deepcopy(used_method_init))
        member_used_method.append(set())
        population.used_method_result = copy.deepcopy(member_used_method)
        self.sorted_index = self.statement_impact(population.sample_slice, population.ori_label)
        line = self.sorted_index[0]
        for _ in range(self.pop_size):
            # line = random.randint(0, n-1)
            if population.line_replaced[line] == []:
                population.used_method.append(copy.deepcopy(member_used_method))
                population.translit_members.append(copy.deepcopy(population.translit_slice))
                population.members.append(copy.deepcopy(population.sample_slice))
                continue
            candidate_info = random.choice(population.line_replaced[line])
            exchanged_line = -1
            population.used_method.append(copy.deepcopy(member_used_method))
            # if population.used_method[-1][line]["exchange"] != -1:
            #     candidate_info = population.line_replaced[population.used_method[-1][line]["exchange"]]
            #     exchanged_line = population.used_method[-1][line]["exchange"]
            used_macro_all = population.used_method[-1][-1]
            translit_slice_tmp = copy.deepcopy(population.translit_slice)
            used_method_tmp = copy.deepcopy(population.used_method[-1][line])
            population_member, update_used_method = self.create_population_member(translit_slice_tmp, used_method_tmp, candidate_info, line, population.ori_label, population.macro_names, used_macro_all)
            if exchanged_line == -1 and update_used_method["exchange"] > 0:
                population.used_method[-1][update_used_method["exchange"]]["exchange"] = line
            population.used_method[-1][-1] = population.used_method[-1][-1].union(update_used_method["macro"])
            population.used_method[-1][line] = update_used_method
            # population.used_method[-1][line]['mutate'] = update_used_method['mutate']
            # population.used_method[-1][line]['add'] = update_used_method['add']
            # population.used_method[-1][line]['merge'] = update_used_method['merge']
            # population.used_method[-1][line]['const'] = update_used_method['const']
            # population.used_method[-1][line]['macro'] = update_used_method['macro']
            population.translit_members.append(population_member)
            population.members.append(self.list_to_statement(population_member))
        dataloader = self.make_batch(population.members)
        _, population.fitness_scores = self.fitness(dataloader, population.ori_label)
        state_score_arr = np.array(population.fitness_scores)
        sort_index = np.argsort(state_score_arr)
        population.result = copy.deepcopy(population.members[sort_index[0]])
        population.translit_result = copy.deepcopy(population.translit_members[sort_index[0]])
        population.used_method_result = copy.deepcopy(population.used_method[sort_index[0]])

    def select(self, population):
        # 轮盘赌选择
        probilities = F.softmax(torch.tensor(population.fitness_scores)).tolist()
        probilities[0] += 1-sum(probilities)
        np.random.seed(int(time.time()))
        index_list = [i for i in range(self.pop_size)]
        selected_index_list = np.random.choice(index_list, size=int(self.pop_size*self.ratio["select"]), p=probilities)
        selected_members = [population.translit_members[i] for i in selected_index_list]
        selected_used_method = [population.used_method[i] for i in selected_index_list]
        return selected_members, selected_used_method

    def cross(self, population):
        # 交叉 等概率选择2*交叉样本数量 两个一组 进行交叉 并更新used_method 同时检查约束
        index_list = [i for i in range(2*self.pop_size)]
        np.random.seed(int(time.time()))
        selected_index_list = np.random.choice(index_list, size=2*int(self.pop_size * self.ratio["cross"]))
        i = 0
        crossed_members, crossed_used_method = [], []
        while i < len(selected_index_list):
            p1 = selected_index_list[i]%self.pop_size
            p2 = selected_index_list[i+1]%self.pop_size
            parent_1 = population.translit_members[p1]
            parent_2 = population.translit_members[p2]
            parent_1_used_method = population.used_method[p1]
            parent_2_used_method = population.used_method[p2]
            cross_location = random.randint(0, len(population.sample_slice)-1)
            if cross_location>0 and parent_1_used_method[cross_location-1]['exchange'] >= 0 or parent_2_used_method[cross_location]['exchange'] >= 0 or parent_2_used_method[cross_location]['const'] > 0 or parent_1_used_method[cross_location]['const'] > 0:
                crossed_members.append(copy.deepcopy(parent_1))
                crossed_used_method.append(parent_1_used_method[:-1])
            else:
                crossed_members.append(parent_1[:cross_location] + parent_2[cross_location:])
                crossed_used_method.append(parent_1_used_method[:cross_location] + parent_2_used_method[cross_location:-1])
            used_macro = set()
            for x in crossed_used_method[-1]:
                used_macro = used_macro.union(x["macro"])
            crossed_used_method[-1].append(copy.deepcopy(used_macro))
            i+=2
        return crossed_members, crossed_used_method

    def mutate(self, population):
        # 等概率选择样本进行变异
        index_list = [i for i in range(self.pop_size)]
        np.random.seed(int(time.time()))
        selected_index_list = np.random.choice(index_list, size=int(self.pop_size * self.ratio["mutate"]))
        n = len(population.sample_slice)
        for i in selected_index_list:
            line = random.randint(0, n-1)

            # self.sorted_index = self.statement_impact(population.sample_slice, population.ori_label)
            # line = self.sorted_index[0]
            # self.sorted_index = self.sorted_index[1:]
            if population.line_replaced[line] == []:
                continue
            candidate_info = random.choice(population.line_replaced[line])
            exchanged_line = -1
            if population.used_method[i][line]["exchange"] != -1:
                if population.line_replaced[population.used_method[i][line]["exchange"]] == []:
                    continue
                candidate_info = random.choice(population.line_replaced[population.used_method[i][line]["exchange"]])
                exchanged_line = population.used_method[i][line]["exchange"]
                if candidate_info[0] == "exchange":
                    continue
            used_macro_all = population.used_method[i][-1]
            translit_member_tmp = copy.deepcopy(population.translit_members[i])
            used_method_tmp = copy.deepcopy(population.used_method[i][line])
            population_member, update_used_method = self.create_population_member(translit_member_tmp, used_method_tmp, candidate_info, line, population.ori_label, population.macro_names, used_macro_all)
            population.translit_members[i] = population_member
            if exchanged_line == -1 and update_used_method["exchange"] > 0:
                population.used_method[i][update_used_method["exchange"]]["exchange"] = line
            population.used_method[i][-1] = population.used_method[i][-1].union(update_used_method["macro"])
            population.used_method[i][line] = update_used_method
            # mutated_members.append(population_member)
            # mutated_used_method.append(population.used_method[i])
        # return mutated_members, mutated_used_method

    def run(self, sample_slice, ori_label, index):
        self.check_model_num = 0
        random.seed(time.time())
        population = Population(self.config, sample_slice, index, ori_label)
        self.initial_population(population)
        # self.check_limits(population)
        label, prob = self.predict_single(population.result)
        if label!=ori_label:
            return True, population.result, label, prob
        for _ in range(self.max_iters):
            selected_members, selected_used_method = self.select(population)
            crossed_members, crossed_used_method = self.cross(population)
            # mutate_members, mutate_used_method = self.mutate(population)
            # population.translit_members = selected_members + crossed_members + mutate_members
            # population.used_method = selected_used_method + crossed_used_method + mutate_used_method
            population.translit_members = selected_members + crossed_members
            population.used_method = selected_used_method + crossed_used_method
            self.mutate(population)
            population.members = []
            for translit_member in population.translit_members:
                population.members.append(self.list_to_statement(translit_member))
            self.check_limits(population)
            dataloader = self.make_batch(population.members)
            _, population.fitness_scores = self.fitness(dataloader, population.ori_label)
            state_score_arr = np.array(population.fitness_scores)
            sort_index = np.argsort(state_score_arr)
            population.result = copy.deepcopy(population.members[sort_index[0]])
            population.translit_result = copy.deepcopy(population.translit_members[sort_index[0]])
            population.used_method_result = copy.deepcopy(population.used_method[sort_index[0]])
            label, prob = self.predict_single(population.result)
            if label != ori_label:
                return True, population.result, label, prob
        label, prob = self.predict_single(population.result)
        return False, population.result, label, prob


class Genetic():
    def __init__(self, config, limits, programs_indexes, model_name='VulDetectModel.pt'):
        self.config = config
        self.programs_indexes = programs_indexes
        self.load_temp_files()
        self.genetic_algorithm = GeneticAlgorithm(config, limits, model_name=model_name)
        self.model_name = model_name
        self.limits = limits

    def load_temp_files(self):
        with open(os.path.join(self.config.temp_path, 'index_label.json'), 'r') as index_label_jsn:
            self.index_label_dict = json.load(index_label_jsn)

    def attack(self):
        count, sucess = 0, 0
        failed = []
        with open(os.path.join(self.config.result_path, "genetic_"+self.model_name+"("+str(self.limits)+")(all)(60,20,0.6,0.6,better_init).csv"), 'w', encoding='utf-8', newline='') as r_csv_f:
            csv_writer = csv.writer(r_csv_f)
            csv_writer.writerow(['index', 'true_label', 'ori_label', 'ori_prob', 'ori_code', 'adv_label', 'adv_prob', 'adv_code', 'query_times'])
            for index in self.programs_indexes:
                # if index!="132691":
                #     continue
                index = str(index)
                with open(os.path.join(self.config.sample_slice_path, index), 'r') as slice_f:
                    sample_slice = slice_f.readlines()
                # with open(os.path.join(self.config.sample_path, index), 'r') as map_f:
                #     map_program = map_f.readlines()
                map_program = self.genetic_algorithm.norm_program(sample_slice)
                ground_label = int(self.index_label_dict[index])
                ori_label, ori_prob = self.genetic_algorithm.predict_single(map_program, norm=False)
                if ori_label != ground_label:
                    sample_slice = concat_statement(sample_slice)
                    csv_writer.writerow([index, ground_label, ori_label, ori_prob, sample_slice, ori_label, ori_prob, sample_slice, self.genetic_algorithm.check_model_num])
                    continue
                count += 1
                # if count < 500:
                #     continue
                is_success, adv_slice, adv_label, adv_prob = self.genetic_algorithm.run(sample_slice, ori_label, index)
                if is_success:
                    sucess += 1
                else:
                    failed.append(index)
                sample_slice = concat_statement(sample_slice)
                adv_slice = concat_statement(adv_slice)
                csv_writer.writerow([index, ground_label, ori_label, ori_prob, sample_slice, adv_label, adv_prob, adv_slice, self.genetic_algorithm.check_model_num])
                print(sucess, count, sucess / count)
            csv_writer.writerow([sucess, count, sucess / count])
            csv_writer.writerow(failed)

