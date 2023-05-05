# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/5/12 8:57
# @Function: data process


import os
import re
import math
import nltk
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from Config.ConfigT import MyConf
from Utils.mapping import *


class DataPipline:
    def __init__(self, config):
        self.config = config
        self.columns = ['data_id', 'SyVCs', 'file_fun', 'program_id', 'types', 'map_code', 'orig_code', 'label']

        if self.config.Norm_symbol:
            self.config.set_value('data', 'data_path', '../resources/Dataset/Map/')
            self.config.set_value('data', 'embedding_path', '../resources/Dataset/Map/embedding')
        else:
            self.config.set_value('data', 'data_path', '../resources/Dataset/NoMap/')
            self.config.set_value('data', 'embedding_path', '../resources/Dataset/NoMap/embedding')


    def load_data_by_path(self, path):
        file_list = os.listdir(path)
        data_ids, SyVCs, files, types, program_ids, orig_code_slices, map_code_slices, labels = [], [], [], [], [], [], [], []
        for file in file_list:
            lines = open(os.path.join(path, file), encoding='UTF-8').readlines()
            SyVC = os.path.splitext(file)[0]
            prev = ''
            instance_index = -1
            temp_list = []
            orig_list = []
            tag = True
            for i, line in enumerate(lines):
                line = line.strip()
                # 1 CVE
                id_match = re.match('\d+', line)
                if id_match and len(line) > 15:
                    strList = line.split(' ')
                    data_id = strList[0]
                    index = strList[1].find('/')
                    program_id = strList[1][:index]
                    file_name = strList[1][index + 1:]
                    if file_name.startswith('CWE'):
                        nameList = file_name.split('_')
                        type = nameList[0]
                    else:
                        type = 'NVD'

                # 2 label
                elif re.match('------------------------------', line):
                    orig_code_slices.append(orig_list[:-1])  # remove the last line of code (label) from temp_list
                    inst_statements = temp_list[:-1]
                    # Perform lexical symbol normalization via the mapping function provided by the SySeVR author
                    map_program, _ = mapping(inst_statements)
                    map_code_slices.append(map_program)
                    data_ids.append(data_id)
                    SyVCs.append(SyVC)
                    labels.append(int(prev))
                    files.append(file_name)
                    program_ids.append(program_id)
                    types.append(type)
                    temp_list = []
                    orig_list = []
                else:
                    # stopwords = set(nltk.corpus.stopwords.words('english'))
                    # temp_list.extend(nltk.word_tokenize(line.lower()))
                    orig_list.append(line)
                    tokens = create_tokens(line)  # tokens: list
                    temp_list.append(tokens)
                prev = line

        data = {'data_id': data_ids, 'SyVCs': SyVCs, 'file_fun': files, 'program_id': program_ids, 'types': types, 'map_code': map_code_slices,
                'orig_code': orig_code_slices, 'label': labels}
        all_data = pd.DataFrame(data, columns=self.columns)
        return all_data


    def load_all(self, data_dir):
        '''
        load all samples
        :param data_dir:
        :return:
        '''
        data = pd.DataFrame(columns=self.columns)
        for t_index, sub_path in enumerate(os.listdir(data_dir)):
            if sub_path not in ['API function call', 'Arithmetic expression', 'Array usage', 'Pointer usage']:
                continue
            data_of_sub = self.load_data_by_path(os.path.join(data_dir, sub_path))
            data = pd.concat([data, data_of_sub])
        data.to_pickle(os.path.join(self.config.data_path, 'data.pkl'))

    def get_split_ids(self, data):
        id_range = data['program_id'].values.tolist()  # the number of slices in data is 420,626
        id_list = list(set(id_range))  # the number of programs in data is 19,113
        program_num = len(id_list)
        print('The total programs of instances in is %s' % (program_num))
        ratios = [int(r) for r in self.config.ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * program_num)
        # seed=113
        np.random.seed(123)
        np.random.shuffle(id_list)
        train_ids = id_list[:train_split]
        train_ids = [str(x) for x in train_ids]
        test_ids = id_list[train_split:]
        test_ids = [str(x) for x in test_ids]
        with open(self.config.data_path + '/../train_ids', 'w') as f:
            json.dump(train_ids, f)
        with open(self.config.data_path + '/../test_ids', 'w') as f:
            json.dump(test_ids, f)

    def split_data(self):
        data = pd.read_pickle(self.config.data_path + '/data.pkl')
        # obtain the program ids for training and testing dataset respectively
        self.get_split_ids(data)

        # obtain train and test
        data_num = len(data)
        print('The total number of instances is %s' % (data_num))
        with open(self.config.data_path + '/../train_ids', 'r') as f:
            train_ids = json.load(f)
        with open(self.config.data_path + '/../test_ids', 'r') as f:
            test_ids = json.load(f)
        train = data[data['program_id'].isin(train_ids)]
        test = data[data['program_id'].isin(test_ids)]

        train_sub = train.sample(n=30000, random_state=123)
        test_sub = test.sample(n=7500, random_state=123)
        print('The total number of train instances is %s' % (len(train_sub)))  # 337465
        print('The total number of test instances is %s' % (len(test_sub)))  # 83161

        def check_or_create(path):
            if not os.path.exists(path):
                os.makedirs(path)

        train_path = self.config.data_path + '/train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train.pkl'
        train_sub.to_pickle(self.train_file_path)

        test_path = self.config.data_path + '/test/'
        check_or_create(test_path)
        self.test_file_path = test_path + 'test.pkl'
        test_sub.to_pickle(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        if not input_file:
            input_file = os.path.join(self.config.data_path, 'train', 'train.pkl')
        data = pd.read_pickle(input_file)
        if not os.path.exists(self.config.embedding_path):
            os.mkdir(self.config.embedding_path)

        def trans_to_sequences(statement_lst):
            sequence = []
            for statement in statement_lst:
                # sequence.extend(nltk.word_tokenize(statement.lower()))
                sequence.extend(create_tokens(statement.lower()))
            return sequence

        if self.config.Norm_symbol:
            corpus = data['map_code'].apply(trans_to_sequences)
        else:
            corpus = data['orig_code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        data['code'] = pd.Series(str_corpus)
        data.to_csv(self.config.embedding_path + '/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=5)
        w2v.save(self.config.embedding_path + '/node_w2v_' + str(size))

    @ staticmethod
    def states2idseq(statement_lst,vocab,max_token):
        '''
        convert code symbols in statement_lst into symbols index in vocab
        :param statement_lst: list, a sequence of statements
        :param vocab:
        :param max_token: the maximum integer index of symbol in vocab +1
         :return:
        '''
        sequence = []
        s_lengths = []
        for index, statement in enumerate(statement_lst):
            # s_split = nltk.word_tokenize(statement.lower())
            s_split = create_tokens(statement.lower())
            sequence.extend([vocab[token].index if token in vocab else max_token for token in s_split])
            s_lengths.append(len(s_split))
        return [sequence, s_lengths]

    # generate block sequences with index representations
    def gen_code_seqs(self, data_path, size):
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.config.embedding_path + '/node_w2v_' + str(size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]
        data = pd.read_pickle(data_path)
        # store vocabulary index and vulnerability locations of tokens
        if self.config.Norm_symbol:
            data['token_indexs_length'] = data.apply(lambda x: self.states2idseq(x['map_code'],vocab,max_token),
                                                     axis=1)
        else:
            data['token_indexs_length'] = data.apply(lambda x: self.states2idseq(x['orig_code'],vocab,max_token),
                                                     axis=1)
        data.to_pickle(os.path.dirname(data_path) + '/blocks.pkl')

def main():
    config = MyConf('../Config/config.cfg')
    pipline = DataPipline(config)
    # 0 vulnerability statistic by type
    # pipline.data_statistic(config.target_programs)

    # 1
    print('load data...')
    data_dir = '../resources/Dataset/Programs/'
    pipline.load_all(data_dir)

    # 2
    print('split the dataset into train and test')
    pipline.split_data()

    # 3
    print('dictionary and embedding...')
    size = 60
    pipline.dictionary_and_embedding(None, size)

    # 4
    print('generate block sequences...')
    pipline.gen_code_seqs(os.path.join(pipline.config.data_path, 'train', 'train.pkl'), size)
    pipline.gen_code_seqs(os.path.join(pipline.config.data_path, 'test', 'test.pkl'), size)


if __name__ == '__main__':
    main()
    # config = MyConf('../Config/config.cfg')
    # pipline = DataPipline(config)
    # 0 vulnerability statistic by type
    # pipline.data_statistic(config.train_programs,'train')
    # data=pd.read_pickle(os.path.join(config.data_path, 'data.pkl'))
    # print(len(data))
    # print('program number statistic')
    # print(data['program_id'].value_counts())
    # print('type number statistic')
    # print(data['types'].value_counts())
