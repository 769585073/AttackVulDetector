# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/5/12 8:57
# @Function: other tool functions
import torch


def evaluate_b(y, y_pred):
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, \
        recall_score, \
        f1_score
    import re
    report_text = classification_report(y, y_pred, target_names=['nsbr', 'sbr'])
    # print(report_text)
    report_list = re.sub(r'[\n\s]{1,}', ' ', report_text).strip().split(' ')
    conf_matrix = confusion_matrix(y, y_pred)
    # print(conf_matrix)
    TN = conf_matrix.item((0, 0))
    FN = conf_matrix.item((1, 0))
    TP = conf_matrix.item((1, 1))
    FP = conf_matrix.item((0, 1))
    prec = 100 * precision_score(y, y_pred, average='binary')
    recall = 100 * recall_score(y, y_pred, average='binary')
    f_measure = 100 * f1_score(y, y_pred, average='binary')
    accuracy = 100 * accuracy_score(y, y_pred)
    return accuracy, prec, recall, f_measure


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    data, program_ids, fine_labels, labels = [], [], [], []
    for _, item in tmp.iterrows():
        data.append(item['token_indexs_length'])
        fine_labels.append(item['label'])
        program_ids.append(item['program_id'])
        if item['label'] == 0:
            labels.append(0)
        else:
            labels.append(1)
        # labels.append(item['label'])
    return data, torch.LongTensor(labels), fine_labels, program_ids


# parameter setting
def adjust_learning_rate(optimizer,learning_rate, epoch):
    lr = learning_rate * (0.5 ** (epoch // 6))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def concat_statement(slice_list):
    r = ''
    for s in slice_list:
        r += s + '<EOL>'
    return r


def write_file():
    with open('../resources/Dataset/test_file.txt','w') as f:
        string='jiangyuan'
        f.write(string)

if __name__ == '__main__':
    write_file()