from Config.ConfigT import MyConf
from gensim.models.word2vec import Word2Vec
from Utils.get_tokens import create_tokens
from Utils.mapping import mapping
from Target_model.RNNDetect import DetectModel
from Target_model.RunNet import RunNet

import numpy as np

import os
import csv
import torch


split_token = '<EOL>'


def norm_program(program_slice):
    # 标准化切片
    inst_statements = []
    for line in program_slice:
        token_list = create_tokens(line)
        inst_statements.append(token_list)
    map_program, _ = mapping(inst_statements)
    return map_program


def b(config, file_name, model_name):
    word2vec = Word2Vec.load(config.embedding_path + "/node_w2v_60").wv
    config.embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    config.embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    config.embedding_dim = word2vec.vectors.shape[1]
    config.vocab_size = word2vec.vectors.shape[0] + 1
    config.vocab = word2vec.vocab

    detect_model = DetectModel(config)
    detect_model.load_state_dict(torch.load(config.models_path + model_name))
    detect_model.cuda()
    run_net = RunNet(config)
    success, skipped = 0, 0
    with open(os.path.join(config.result_path, file_name), 'r') as csv_f:
        reader = csv.reader(csv_f)
        content = list(reader)
        for line in content[1:1001]:
            index, ground_label, ori_label, ori_prob, ori_slice, adv_label, adv_prob, adv_slice, query_times = line
            ori_slice_list = ori_slice.split(split_token)
            ori_norm = norm_program(ori_slice_list)
            ori_label, ori_prob = run_net.predict_single(detect_model, ori_norm)
            if int(ground_label) != int(ori_label):
                skipped+=1
                continue
            adv_slice_list = adv_slice.split(split_token)
            adv_norm = norm_program(adv_slice_list)
            label, prob = run_net.predict_single(detect_model, adv_norm)
            if label!=int(ground_label):
                success+=1
            print(skipped, success)
        print(success / (1000 - skipped))



if __name__=="__main__":
    config = MyConf('../Config/config.cfg')
    file_name = "combination_VulDetectModel.pt(all,greedy,15).csv"
    model_name = "adv_VulDetectModel_1.0.pt"
    # model_name = "VulDetectModel.pt"
    b(config, file_name, model_name)