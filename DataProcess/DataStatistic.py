# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/5/15 8:51
# @Function:
from Config.ConfigT import MyConf
import pandas as pd
import os

def num_pos_neg(data):
    pos=data[data['label']==1]
    print('The number of positive samples is %s, ratio is %f'  % (len(pos),len(pos)/len(data)))

    neg=data[data['label']==0]
    print('The number of negative samples is %s, ratio is %f'  % (len(neg),len(neg)/len(data)))

    NVD = data[data['types']=='NVD']
    print('The number of NVD samples is %s, ratio is %f' % (len(NVD), len(NVD) / len(data)))

def statis_dataset(config):
    data = pd.read_pickle(os.path.join(config.data_path, 'data.pkl'))
    print(len(data))
    print('program number statistic')
    print(data['program_id'].value_counts())
    print('type number statistic')
    print(data['types'].value_counts())



def train_test_statis():
    train_data = pd.read_pickle(config.data_path + 'train/blocks.pkl')
    test_data = pd.read_pickle(config.data_path + 'test/blocks.pkl')
    print('Positive')
    num_pos_neg(train_data)
    print('Negative')
    num_pos_neg(test_data)


# def test_compare():
#     path = config.data_path
#     map = path
#     _map = path+ '../_Map/'
#     test_map = map + '/test/test.pkl'
#     test_map_ = _map + '/test/test.pkl'
#     test_map_data = pd.read_pickle(test_map)
#     test_map_data_ = pd.read_pickle(test_map_)
#
#     print(len(test_map_data))
#     print(len(test_map_data_))




if __name__ == '__main__':
    config = MyConf('../Config/config.cfg')
    train_test_statis()
    # test_compare()