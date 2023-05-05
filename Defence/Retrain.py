import os
import csv
import json
import warnings
import random
import pandas as pd
import numpy as np
from Config.ConfigT import MyConf
from Utils.get_tokens import create_tokens
from Utils.mapping import mapping
from Target_model.RunNet import RunNet
from gensim.models.word2vec import Word2Vec
from DataProcess.DataPipline import DataPipline
from Entry.main import Run_Entry

from Attack.Mutation import Mutation
from Attack.Obfuscation import Obfuscation
from Attack.Combination import Combination
from Attack.Genetic import Genetic
from Attack.AttackTarget import AttackTarget


warnings.filterwarnings("ignore")


class MakeRetrainData():
    def __init__(self, config):
        self.config = config
        self.split_token = '<EOL>'
        self.tool = DataPipline(config)
        self.entry = Run_Entry(config)

    def norm_program(self, program_slice):
        if program_slice == None:
            return []
        inst_statements = []
        for line in program_slice:
            token_list = create_tokens(line)
            inst_statements.append(token_list)
        map_program, _ = mapping(inst_statements)
        return map_program

    def process_slice(self, slice):
        slice_statements = slice.split(self.split_token)
        statements = []
        for statement in slice_statements:
            if not statement.endswith('\n'):
                statement = statement+'\n'
            statements.append(statement)
        return statements

    def create_attack_samples(self):
        train_data = pd.read_pickle(config.data_path + 'train/blocks.pkl')
        sample_ids_json = os.path.join(config.root_path, 'sample_ids.json')
        sample_tag = False
        create_vul_line_number_tag = False
        creat_token_importance_tag = False
        if sample_tag:
            programs = self.entry.sample_Test(train_data)
            self.entry.create_benchmark()
        with open(sample_ids_json, 'r') as sample_ids_file:
            programs_indexes = json.load(sample_ids_file)
        if create_vul_line_number_tag:
            self.entry.create_line_number(programs_indexes)
            self.entry.create_vul_line_number(programs_indexes)
        if creat_token_importance_tag:
            c = Combination(self.config, 0, programs_indexes)
            c.get_candidates_by_token_importance([], 0)

        # attack = AttackTarget(config)
        # target_model = attack.load_trained_model('VulDetectModel.pt')
        # mutation = Mutation(config, programs_indexes, target_model)
        # mutation.attack()

        # replace_limit_number = 1
        # obfuscation = Obfuscation(config, replace_limit_number, programs_indexes)
        # obfuscation.macro_raplace_attack()
        # obfuscation = Obfuscation(config, replace_limit_number, programs_indexes)
        # obfuscation.replace_line_attack()
        # obfuscation = Obfuscation(config, replace_limit_number, programs_indexes)
        # obfuscation.add_dead_code_attack()
        # obfuscation = Obfuscation(config, replace_limit_number, programs_indexes)
        # obfuscation.replace_const_attack()
        # obfuscation = Obfuscation(config, replace_limit_number, programs_indexes)
        # obfuscation.merge_function_attack()

        # replace_limit_number = 14
        # combination = Combination(config, replace_limit_number, programs_indexes)
        # combination.combination_attack(random_tag=True)

        # limits = 14
        # genetic = Genetic(config, limits, programs_indexes)
        # genetic.attack()

    def main(self, file_names, ratio=1.0, only_sucess=True):
        print("creating retrain data set...")
        ori_train_data = pd.read_pickle(config.data_path + 'train/blocks.pkl')
        # ori_test_data = pd.read_pickle(config.data_path + 'test/blocks.pkl')
        columns = ['data_id', 'SyVCs', 'file_fun', 'program_id', 'types', 'map_code', 'orig_code', 'label', 'token_indexs_length']
        data_ids, SyVCs, files, program_ids, types, map_code_slices, orig_code_slices, labels, token_indexs_length = [], [], [], [], [], [], [], [], []
        for file_name in file_names:
            with open(os.path.join(config.result_path, file_name), 'r') as csv_f:
                reader = csv.reader(csv_f)
                content = list(reader)
                for line in content[1:1001]:
                    index, ground_label, ori_label, ori_prob, ori_slice, adv_label, adv_prob, adv_slice, query_times = line
                    if only_sucess and ori_label==adv_label:
                        continue

                    adv_slice = self.process_slice(adv_slice)
                    adv_slice_norm = self.norm_program(adv_slice)

                    data_ids.append(ori_train_data.loc[int(index)]['data_id'])
                    SyVCs.append(ori_train_data.loc[int(index)]['SyVCs'])
                    files.append(ori_train_data.loc[int(index)]['file_fun'])
                    program_ids.append(ori_train_data.loc[int(index)]['program_id'])
                    types.append(ori_train_data.loc[int(index)]['types'])
                    map_code_slices.append(adv_slice_norm)
                    orig_code_slices.append(adv_slice)
                    labels.append(int(ground_label))
                    if self.config.Norm_symbol:
                        token_indexs_length.append(self.tool.states2idseq(adv_slice_norm,self.config.vocab,self.config.vocab_size-1))
                    else:
                        token_indexs_length.append(self.tool.states2idseq(adv_slice, self.config.vocab, self.config.vocab_size-1))
        data = {'data_id': data_ids, 'SyVCs': SyVCs, 'file_fun': files, 'program_id': program_ids, 'types': types, 'map_code': map_code_slices, 'orig_code': orig_code_slices, 'label': labels, 'token_indexs_length': token_indexs_length}
        adv_data = pd.DataFrame(data, columns=columns)
        if only_sucess:
            fine_tuning_data_path = os.path.join(self.config.defence_path, 'only_success_fine_tuning_data_' + str(ratio) + '.pkl')
        else:
            fine_tuning_data_path = os.path.join(self.config.defence_path, 'fine_tuning_data_' + str(ratio) + '.pkl')
        adv_data_sample = adv_data.sample(n=int(ratio*len(adv_data)), random_state=np.random.RandomState())
        adv_data_sample.to_pickle(fine_tuning_data_path)
        all_data = pd.concat((ori_train_data, adv_data_sample), axis=0)
        if only_sucess:
            adv_data_path = os.path.join(self.config.defence_path, 'only_success_adv_train_data_' + str(ratio) + '.pkl')
        else:
            adv_data_path = os.path.join(self.config.defence_path, 'adv_train_data_' + str(ratio) + '.pkl')
        all_data.to_pickle(adv_data_path)
        print("Done!")


if __name__ == '__main__':
    ratio = 1.0
    double_adv_samples = False
    retrain = True
    extend_advs = False
    make_data = True
    fine_tuning = False
    only_sucess = False
    config = MyConf('../Config/config_defence.cfg')
    ori_train_data = pd.read_pickle(config.data_path + 'train/blocks.pkl')
    word2vec = Word2Vec.load(config.embedding_path + "/node_w2v_60").wv
    config.embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    config.embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    config.embedding_dim = word2vec.vectors.shape[1]
    config.vocab_size = word2vec.vectors.shape[0] + 1
    config.vocab = word2vec.vocab

    mrd = MakeRetrainData(config)
    if extend_advs:
        mrd.create_attack_samples()

    if retrain:
        if make_data:
            file_names = []
            for file in os.listdir(config.result_path):
                file_names.append(file)
            mrd.main(file_names, ratio=ratio, only_sucess=only_sucess)

        if fine_tuning:
            if only_sucess:
                train_data = pd.read_pickle(os.path.join(config.defence_path, 'only_success_fine_tuning_data_' + str(ratio) + '.pkl'))
                model_name = 'only_success_fine_tuning_VulDetectModel_' + str(ratio) + '.pt'
            else:
                train_data = pd.read_pickle(os.path.join(config.defence_path, 'fine_tuning_data_' + str(ratio) + '.pkl'))
                model_name = 'fine_tuning_VulDetectModel_' + str(ratio) + '.pt'
        else:
            if only_sucess:
                train_data = pd.read_pickle(os.path.join(config.defence_path, 'only_success_adv_train_data_' + str(ratio) + '.pkl'))
                model_name = 'only_success_adv_VulDetectModel_' + str(ratio) + '.pt'
            else:
                train_data = pd.read_pickle(os.path.join(config.defence_path, 'adv_train_data_' + str(ratio) + '.pkl'))
                model_name = 'adv_VulDetectModel_' + str(ratio) + '.pt'

        if double_adv_samples:
            add_data = pd.read_pickle(os.path.join(config.defence_path, "fine_tuning_data_1.0.pkl"))
            train_data = pd.concat((train_data, add_data), axis=0)
            model_name = 'adv_VulDetectModel_' + str(2.0) + '.pt'

        train_data = train_data.sample(frac=1)
        test_data = pd.read_pickle(config.data_path + 'test/blocks.pkl')
        print(len(train_data), len(ori_train_data))
        run_net = RunNet(config)
        run_net.train_eval_model(train_data, test_data, model_name, fine_tuning=fine_tuning)

    # 复现最开始的模型 加了正交初始化 效果好于最开始的模型 主要是为了每个epoch的指标
    # train_data = pd.read_pickle(config.data_path + 'train/blocks.pkl')
    # test_data = pd.read_pickle(config.data_path + 'test/blocks.pkl')
    # model_name = 'recurrent_VulDetectModel.pt'
    # run_net = RunNet(config)
    # run_net.train_eval_model(train_data, test_data, model_name)

