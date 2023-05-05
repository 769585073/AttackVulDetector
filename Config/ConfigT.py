# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/5/12 13:57
# @Function: configure file

from configparser import ConfigParser


class MyConf:
    def __init__(self, config_file):
        config = ConfigParser()
        config.read(config_file, encoding='utf-8')
        self._config = config
        self.config_file=config_file

        # config.write(open(config_file,'w'))

    @property
    def wordEmbedding(self):
        return self._config.getboolean('data', 'word_embedding')

    @property
    def charData(self):
        return self._config.getboolean('data', 'char_data')

    @property
    def train_programs(self):
        return self._config.get('data', 'train_programs')

    @property
    def target_programs(self):
        return self._config.get('data', 'target_programs')

    @property
    def data_path(self):
        return self._config.get('data', 'data_path')

    @property
    def sample_path(self):
        return self._config.get('data', 'sample_path')

    @property
    def ratio(self):
        value = self._config.get("data", "train_test_ratio")
        # print(list(value))
        return value

    @property
    def Norm_symbol(self):
        return self._config.getboolean('data', 'Norm_symbol')

    @property
    def models_path(self):
        return self._config.get('data', 'models_path')

    @property
    def embedding_path(self):
        return self._config.get('data', 'embedding_path')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def weight_decay(self):
        return self._config.getfloat('Optimizer', 'weight_decay')

    @property
    def batch_size(self):
        return self._config.getint('params', 'batch_size')

    @property
    def class_num(self):
        return self._config.getint('params', 'class_num')

    @property
    def lstm_hidden_dim(self):
        return self._config.getint('params', 'lstm_hidden_dim')

    @property
    def lstm_num_layers(self):
        return self._config.getint('params', 'lstm_num_layers')

    @property
    def embedding_ast_retrain(self):
        return self._config.getboolean('params', 'embedding_ast_retrain')

    @property
    def batch_normalizations(self):
        return self._config.getboolean('params', 'batch_normalizations')

    @property
    def bath_norm_momentum(self):
        return self._config.getfloat("params", "bath_norm_momentum")

    @property
    def batch_norm_affine(self):
        return self._config.getboolean("params", "batch_norm_affine")

    @property
    def dropout(self):
        return self._config.getfloat("params", "dropout")

    @property
    def dropout_embed(self):
        return self._config.getfloat("params", "dropout_embed")

    @property
    def kernel_num(self):
        return self._config.getint("params", "kernel_num")

    @property
    def kernel_sizes(self):
        value = self._config.get("params", "kernel_sizes")
        # print(list(value))
        value = [int(k) for k in list(value) if k != ","]
        return value

    @property
    def max_sen_length(self):
        return self._config.getint('params', 'max_sen_length')

    @property
    def init_weight(self):
        return self._config.getboolean("params", "init_weight")

    @property
    def init_weight_value(self):
        return self._config.getfloat("params", "init_weight_value")

    @property
    def samplecnt(self):
        return self._config.getint('RFParams', 'samplecnt')

    @property
    def epsilon(self):
        return self._config.getfloat('RFParams', 'epsilon')


    @property
    def alpha(self):
        return self._config.getfloat('RFParams', 'alpha')

    @property
    def tau(self):
        return self._config.getfloat('RFParams', 'tau')

    @property
    def delay_critic(self):
        return self._config.getboolean('RFParams', 'delay_critic')

    # Train
    @property
    def epochs(self):
        return self._config.getint("Train", "epochs")

    # use
    @property
    def use_plot(self):
        return self._config.getboolean('use', 'use_plot')

    @property
    def use_save(self):
        return self._config.getboolean('use', 'use_save')

    @property
    def use_gpu(self):
        return self._config.getboolean('use', 'use_gpu')

    @property
    def sample_source_path(self):
        return self._config.get('data', 'sample_source_path')

    @property
    def source_path(self):
        return self._config.get('data', 'source_path')

    @property
    def temp_path(self):
        return self._config.get('data', 'temp_path')

    @property
    def sample_slice_path(self):
        return self._config.get('data', 'sample_slice_path')

    @property
    def result_path(self):
        return self._config.get('data', 'result_path')

    @property
    def mutation_path(self):
        return self._config.get('data', 'mutation_path')

    @property
    def slice_with_line_path(self):
        return self._config.get('data', 'slice_with_line_path')

    @property
    def vul_info_path(self):
        return self._config.get('data', 'vul_info_path')

    @property
    def defence_path(self):
        return self._config.get('data', 'defence_path')

    @property
    def root_path(self):
        return self._config.get('data', 'root_path')

    @property
    def log_path(self):
        return self._config.get('data', 'log_path')

    def set_value(self, section, param, value):
        self._config.set(section, param, value)
        self._config.write(open(self.config_file, 'w'))

if __name__ == '__main__':
    conf = MyConf('config.cfg')
    print(conf.wordEmbedding)
    print(conf.charData)
