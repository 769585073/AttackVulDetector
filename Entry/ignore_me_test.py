import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from gensim.models.word2vec import Word2Vec
from DataProcess.DataPipline import DataPipline
from Target_model.RNNDetect import DetectModel
from Config.ConfigT import MyConf
from Attack.AttackTarget import AttackTarget
from Utils.get_tokens import create_tokens

def make_batch(sample_ids, vocab, max_token, batch_size = 64):
    dataset = []
    batch = []
    for i, index in enumerate(sample_ids):
        if i%batch_size == 0 and batch!=[]:
            dataset.append(batch)
            batch = []
        map_program_path = os.path.join("../resources/Dataset/Samples", str(index))
        with open(map_program_path, 'r') as map_program_f:
            map_program = map_program_f.readlines()
        input_seq = DataPipline.states2idseq(map_program, vocab, max_token-1)
        batch.append(input_seq)
    if batch!=[]:
        dataset.append(batch)
    return dataset


def run_model_by_batch(dataloader, model):
    predicted = []
    probability= []
    for batch in dataloader:
        model.batch_size = len(batch)
        logits = model(batch)
        softs = F.softmax(logits, dim=1)
        preds = torch.max(softs, 1)[1]
        probs = torch.index_select(softs, 1, preds).t()[0]
        predicted.extend(preds.cpu().data.numpy().tolist())
        probability.extend(probs.cpu().data.numpy().tolist())
    return predicted, probability

def main():
    word2vec = Word2Vec.load(config.embedding_path + "/node_w2v_60").wv
    config.embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    config.embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    config.embedding_dim = word2vec.vectors.shape[1]
    config.vocab_size = word2vec.vectors.shape[0] + 1
    config.vocab = word2vec.vocab

    attack = AttackTarget(config)
    model = attack.load_trained_model('VulDetectModel.pt')
    # model_path = '../resources/SavedModels/VulDetectModel.pt'
    # model = torch.load(model_path)
    # model.cuda()
    sample_ids_path = "../resources/Dataset/sample_ids.json"
    with open(sample_ids_path, 'r') as sample_ids_jsn:
        sample_ids = json.load(sample_ids_jsn)
    dataloader = make_batch(sample_ids, config.vocab, config.vocab_size)
    preds, probs = run_model_by_batch(dataloader, model)
    print(probs)

    label_path = os.path.join(config.temp_path, 'index_label.json')
    with open(label_path, 'r') as label_jsn:
        index_labels = json.load(label_jsn)
    count = 0
    for i, index in enumerate(sample_ids):
        pred = preds[i]
        prob = probs[i]
        ground = index_labels[str(index)]
        print(index, pred, ground, prob)


def count_while_samples(config):
    has_while_count = 0
    has_bool_count = 0
    has_for_count = 0
    has_do_count = 0
    has_total_recr = 0
    for sample in os.listdir(config.sample_slice_path):
        sample_path = os.path.join(config.sample_slice_path, sample)
        print(sample_path)
        with open(sample_path, 'r', encoding='utf-8', errors='ignore') as f:
            slice = f.readlines()
        for line in slice:
            token_list = create_tokens(line)
            if "while" in token_list:
                has_while_count += 1
                print("while:", has_while_count)
                print("while:", sample)
                break
        # for line in slice:
        #     token_list = create_tokens(line)
        #     if "false" in token_list or "true" in token_list or "False" in token_list or "True" in token_list or "FALSE" in token_list or "TRUE" in token_list :
        #         has_bool_count += 1
        #         print("bool:", has_bool_count)
        #         print("bool:", sample)
        #         break

        for line in slice:
            token_list = create_tokens(line)
            if "do" in token_list:
                has_do_count += 1
                print("do:", has_do_count)
                print("do:", sample)
                break

        for line in slice:
            token_list = create_tokens(line)
            if "for" in token_list:
                has_for_count += 1
                print("for:", has_for_count)
                print("for:", sample)
                print(slice)
                break

        for line in slice:
            token_list = create_tokens(line)
            if "while" in token_list or "for" in token_list or "do" in token_list:
                has_total_recr += 1
                print("total_recr:", has_total_recr)
                print("total_recr:", sample)
                break

    print("while_total: ", has_while_count)
    print("do_total: ", has_do_count)
    print("for_total: ", has_for_count)
    print("all_recr_total: ", has_total_recr)


def count_adv_samples_ratio(config):
    ratio = 0.1
    train_data = pd.read_pickle(os.path.join(config.defence_path, 'fine_tuning_data_' + str(ratio) + '.pkl'))
    print(len(train_data))


def random_samples_test(ratio = 0.1):
    n = 15
    ids = [[i]*1000 for i in range(n)]
    ids_new = []
    for id in ids:
        ids_new.extend(id)
    data = {'ids': ids_new}
    columns = ['ids']
    data_frame = pd.DataFrame(data, columns=columns)
    random_data_sample = data_frame.sample(n=int(ratio * len(data_frame)), random_state=np.random.RandomState())
    random_samples_list = [len(random_data_sample[random_data_sample['ids']==i]) for i in range(n)]
    # print(random_samples_list)
    return max(random_samples_list)-min(random_samples_list)


if __name__ == '__main__':
    config = MyConf('../Config/config.cfg')
    # main()
    # count_while_samples(config)
    # count_adv_samples_ratio(config)
    ratios = [(i+1)/10 for i in range(10)]
    dis_ratio = []
    for ratio in ratios:
        print(ratio)
        dis = []
        for _ in range(10000):
            dis.append(random_samples_test())
        dis_ratio.append(dis)
        print(max(dis))
    for ratio, dis in zip(ratios, dis_ratio):
        print(ratio, ' : ', max(dis))