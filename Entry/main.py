# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/5/12 8:50
# @Function:
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from gensim.models.word2vec import Word2Vec

from AttacVulDecModel.Utils.mapping import *
from Target_model.RunNet import RunNet
from Attack.AttackTarget import AttackTarget
from Config.ConfigT import MyConf
from xml.dom.minidom import parse
from Utils.nope import *

import shutil, sys
from Attack.Mutation import Mutation
from Attack.Obfuscation import Obfuscation
from Attack.Combination import Combination
from Attack.Genetic import Genetic


class Run_Entry:
    def __init__(self, config):
        self.config = config
        self.loss_fn = F.cross_entropy
        self.runnet = RunNet(config)  # [train eval predict] target model

    def train_target_model(self, train_data, test_data, model_name='VulDetectModel.pt'):
        tag = self.runnet.train_eval_model(train_data, test_data, model_name)
        if tag:
            print('Target model has been trained!')

    def sample_Test(self, test_data):
        test_pos = test_data[test_data['label'] == 1]
        test_neg = test_data[test_data['label'] == 0]
        sample_path = self.config.sample_path
        samples_pos = test_pos.sample(n=500, random_state=1)
        samples_neg = test_neg.sample(n=500, random_state=1)
        samples = pd.concat([samples_pos, samples_neg])
        samples_program_id = self.process_samples(samples, sample_path)
        self.find_samples_location(samples_program_id)
        return samples

    def process_samples(self, samples, sample_path):
        # 把 index-id index-preid index-function_named等 做成json存起来 把map_code orig_slice 存在文件中
        tmp_path = self.config.temp_path
        sample_slices_path = self.config.sample_slice_path
        index_id = {}
        sample_indexes = []
        samples_program_id = []
        pre_ids = {}
        index_func = {}
        index_label = {}
        index_dataid = {}
        index_vultype = {}
        for index, row in samples.iterrows():
            index_label[index] = row['label']
            index_func[index] = row['file_fun']
            index_vultype[index] = row['SyVCs']
            program_id = row['program_id']
            index_dataid[index] = row['data_id'] + ' ' + row['program_id'] + '/' + row['file_fun']
            pre_id = self.get_pre_id(program_id)
            if pre_id == None:
                print(program_id, "not in SARD+NVD!!!")
            else:
                pre_ids[index] = pre_id

            sample_indexes.append(str(index))
            index_id[index] = program_id
            orig_code = row['orig_code']
            samples_program_id.append(program_id)
            # mapping
            inst_statements = []
            for line in orig_code:
                token_list = create_tokens(line)
                inst_statements.append(token_list)
            map_program, _ = mapping(inst_statements)

            with open(os.path.join(sample_path, str(index)), 'w') as f:
                for line in map_program:
                    f.writelines(line+'\n')

            with open(os.path.join(sample_slices_path, str(index)), 'w') as slice_f:
                for line in orig_code:
                    slice_f.writelines(line+'\n')

        index_id_json = 'index_id.json'
        with open(os.path.join(tmp_path, index_id_json), 'w') as jsn:
            json.dump(index_id, jsn)

        sample_ids_json = os.path.join(self.config.root_path, 'sample_ids.json')
        with open(sample_ids_json, 'w') as sjsn:
            json.dump(sample_indexes, sjsn)

        index_pre_id_json = 'index_pre_id.json'
        with open(os.path.join(tmp_path, index_pre_id_json), 'w') as pre_id_jsn:
            json.dump(pre_ids, pre_id_jsn)

        index_funcname = 'index_funcname.json'
        with open(os.path.join(tmp_path, index_funcname), 'w') as funcname_jsn:
            json.dump(index_func, funcname_jsn)

        index_label_json = 'index_label.json'
        with open(os.path.join(tmp_path, index_label_json), 'w') as label_jsn:
            json.dump(index_label, label_jsn)

        index_dataid_json = 'index_dataid.json'
        with open(os.path.join(tmp_path, index_dataid_json), 'w') as dataid_jsn:
            json.dump(index_dataid, dataid_jsn)

        index_vul_type = 'index_vultype.json'
        with open(os.path.join(tmp_path, index_vul_type), 'w') as vultype_jsn:
            json.dump(index_vultype, vultype_jsn)

        print('Sample Down!')
        return samples_program_id

    def find_samples_location(self, samples_program_id):
        # 把sample出来的program的源文件放到sample_source_path中
        source_code_path = self.config.source_path
        source_code_samples_path = self.config.sample_source_path + '/'
        file_dirs = os.listdir(source_code_path)
        finded = []
        count = 0
        for file_dir in file_dirs:
            curr_source_code_path = source_code_path + '/' + file_dir + '/'
            program_ids = os.listdir(curr_source_code_path)
            for program_id in program_ids:
                for sample_program_id in samples_program_id:
                    if sample_program_id == program_id:
                        try:
                            source = os.path.abspath(curr_source_code_path + program_id)
                            target = os.path.abspath(source_code_samples_path + program_id)
                            if not os.path.exists(target):
                                os.mkdir(target)
                            shutil.copytree(source, target, dirs_exist_ok=True)
                        except IOError as e:
                            print("Unable to copy file. %s" % e)
                        except:
                            print("Unexpected error:", sys.exc_info())
                        finded.append(sample_program_id)
                        count += 1
        # 看看有没有源文件不存在的program_id
        for x in samples_program_id:
            if x not in finded:
                print(x)
        # print(len(samples_program_id), count)

    def get_pre_id(self, id):
        source_code_path = self.config.source_path
        file_dirs = os.listdir(source_code_path)
        for file_dir in file_dirs:
            if id in os.listdir(os.path.join(source_code_path, file_dir)):
                return file_dir
        return None

    def test_multiple_program(self, detect_model, test_data):
        # test_pos = test_data[test_data['label'] == 1]
        # test_NVD = test_data[test_data['types'] == 'NVD']
        # test_NVD_pos = test_NVD[test_NVD['label'] == 1]
        # test_NVD_neg = test_NVD[test_NVD['label'] == 0]
        test_result, _ = self.runnet.eval_model(detect_model, test_data)
        return test_result

    def test_single_program(self, detect_model, program_path):
        '''
        predicted whether program in program_path (location) is a vulnerability or not
        :param detect_model: trained model
        :param program_path: location
        :return:
        '''
        if os.path.exists(program_path):
            program = open(program_path, encoding='UTF-8').readlines()
            predicted_label, probability = self.runnet.predict_single(detect_model, program)
            print('%s program is predicted label %s with probability of %s' % (
                program_path, predicted_label, probability))

    def get_vul_statement(self, line_numbers, curr_id, file_name):
        line_statement = {}
        if curr_id not in os.listdir(self.config.sample_source_path):
            return line_statement
        if file_name in os.listdir(os.path.join(self.config.sample_source_path, curr_id)):
            with open(os.path.join(self.config.sample_source_path, curr_id, file_name), 'r') as f:
                file_content = f.readlines()
            for line in line_numbers:
                if int(line) > 0:
                    line_statement[line] = file_content[int(line) - 1]
                else:
                    line_statement[line] = None
        return line_statement

    def create_vul_line_number(self, programs_indexes):
        # 为每个index生成一个漏洞行号 行号是切片行号
        index_id_path = os.path.join(self.config.temp_path, "index_id.json")
        with open(index_id_path, 'r') as index_id_jsn:
            index_id_dict = json.load(index_id_jsn)

        index_line_number_path = os.path.join(self.config.temp_path, 'index_line_number.json')
        with open(index_line_number_path, 'r') as index_line_number_jsn:
            index_line_number_dict = json.load(index_line_number_jsn)

        xml_path = '../resources/Dataset/fine_label/SARD_testcaseinfo.xml'
        domTree = parse(xml_path)
        rootNode = domTree.documentElement
        test_cases = rootNode.getElementsByTagName("testcase")
        test_cases_info_dict = {}
        for test_case in test_cases:
            if test_case.hasAttribute("id"):
                id = test_case.getAttribute("id")
                test_case_info = {}
                file_paths = test_case.getElementsByTagName("file")
                # 遍历漏洞对应的文件
                for file_path in file_paths:
                    flaw_lines = file_path.getElementsByTagName("flaw")
                    line_numbers = []
                    # 遍历漏洞文件对应的漏洞行
                    for flaw_line in flaw_lines:
                        line = flaw_line.getAttribute('line')
                        # 漏洞行号
                        line_numbers.append(line)
                    mix_lines = file_path.getElementsByTagName('mixed')
                    for mix_line in mix_lines:
                        line = mix_line.getAttribute('line')
                        # 漏洞行号
                        line_numbers.append(line)
                    file_name = file_path.getAttribute('path').split('/')[-1]
                    line_statement = self.get_vul_statement(line_numbers, id, file_name)
                    test_case_info[file_path.getAttribute('path')] = line_statement
                test_cases_info_dict[id] = test_case_info

        index_vul_line_number_dict = {}
        for index in programs_indexes:
            curr_id = index_id_dict[str(index)]
            line_dict = index_line_number_dict[str(index)]
            if curr_id in test_cases_info_dict.keys():
                test_case_info = test_cases_info_dict[curr_id]
                with open(os.path.join(self.config.sample_slice_path, index), 'r') as program_slice_f:
                    program_slice = program_slice_f.readlines()
                vul_lines_list = []
                for slice_line in range(len(program_slice)):
                    slice_line = str(slice_line)
                    ori_line = str(line_dict[slice_line][0])
                    curr_file = line_dict[slice_line][1]
                    if ori_line == "-1" or curr_file == None:
                        # 没得到行号的行不能进行变异， 按照漏洞行处理
                        vul_lines_list.append(slice_line)
                        continue

                    potential_line_and_content = {}
                    with open(os.path.join(self.config.sample_source_path, curr_id, curr_file), 'r') as vul_source_f:
                        source_content = vul_source_f.readlines()
                    # 注释行以/* POTENTIAL FLAW开头 往下找 找到第一个没有以*开头的语句就是漏洞语句
                    for i, line in enumerate(source_content):
                        if line.strip().startswith('/* POTENTIAL FLAW') or line.strip().startswith('/* FLAW'):
                            j = i + 1
                            while source_content[j].strip().startswith('*') and j < len(source_content):
                                j += 1
                            if j < len(source_content):
                                # 切片中的行号就是以1开始的 所以j+1
                                potential_line_and_content[str(j+1)] = source_content[j]
                    if ori_line in potential_line_and_content.keys():
                        if "".join(program_slice[int(slice_line)].split()) in "".join(potential_line_and_content[ori_line].split()):
                            vul_lines_list.append(slice_line)

                    for key in test_case_info.keys():
                        vul_info_line_statement_dict = test_case_info[key]
                        if ori_line in vul_info_line_statement_dict.keys():
                            compared_content = ""
                            if vul_info_line_statement_dict[ori_line] != None:
                                compared_content = "".join(vul_info_line_statement_dict[ori_line].split())
                            if "".join(program_slice[int(slice_line)].split()) in compared_content:
                                vul_lines_list.append(slice_line)
                vul_lines_list = list(set(vul_lines_list))
                index_vul_line_number_dict[index] = vul_lines_list
            else:
                index_vul_line_number_dict[index] = []
                print(index, curr_id, "not in xml!")
        index_vul_line_number_path = os.path.join(self.config.temp_path, 'index_vul_line_number_dict.json')
        with open(index_vul_line_number_path, 'w') as index_vul_line_number_jsn:
            json.dump(index_vul_line_number_dict, index_vul_line_number_jsn)

    def create_benchmark(self):
        # 生成data_id与benchmark对应关系
        programs_path = '../resources/Dataset/Programs'
        vul_types = os.listdir(programs_path)
        end_line = '------------------------------\n'
        dataid_benchmark = {}
        for type in vul_types:
            file_name = os.listdir(os.path.join(programs_path, type))
            file_path = os.path.join(programs_path, type, file_name[0])
            with open(file_path, 'r') as f:
                contents = f.read()
            contents_list = contents.split(end_line)
            for content in contents_list:
                lines = content.split('\n')
                infomations = lines[0].split()
                if len(infomations) > 2:
                    dataid = ' '.join(infomations[:2])
                    # print(dataid)
                    benchmark = ' '.join(infomations[2:])
                    if dataid in dataid_benchmark.keys():
                        print(dataid, benchmark, dataid_benchmark[dataid])
                    dataid_benchmark[dataid] = benchmark
        dataid_benchmark_path = os.path.join(self.config.temp_path, 'dataid_benchmark_dict.json')
        with open(dataid_benchmark_path, 'w') as dataid_benchmark_jsn:
            json.dump(dataid_benchmark, dataid_benchmark_jsn)

    def create_line_number(self, program_ids):
        # 采用切片基准的方法生成行号 生成{index：{slice_line_number:[source_line_number, file_name]}}
        dataid_benchmark_path = os.path.join(self.config.temp_path, 'dataid_benchmark_dict.json')
        with open(dataid_benchmark_path, 'r') as dataid_benchmark_jsn:
            dataid_benchmark_dict = json.load(dataid_benchmark_jsn)

        index_id_path = os.path.join(self.config.temp_path, 'index_id.json')
        with open(index_id_path, 'r') as index_id_jsn:
            index_id_dict = json.load(index_id_jsn)

        index_dataid_path = os.path.join(self.config.temp_path, 'index_dataid.json')
        with open(index_dataid_path, 'r') as index_dataid_jsn:
            index_dataid_dict = json.load(index_dataid_jsn)

        index_preid_path = os.path.join(self.config.temp_path, 'index_pre_id.json')
        with open(index_preid_path, 'r') as pid_jsn:
            pre_id_dict = json.load(pid_jsn)

        index_vultype_path = os.path.join(self.config.temp_path, 'index_vultype.json')
        with open(index_vultype_path, 'r') as index_vultype_jsn:
            index_vultype_dict = json.load(index_vultype_jsn)

        type_dict = {'Array usage': 'arrayslice', 'Pointer usage': 'pointslice', 'API function call': 'apislice',
                     'Arithmetic expression': 'integeslice'}
        end_line = '------------------------------\n'

        index_line_dict = {}
        count = 0
        for index in program_ids:
            count += 1
            print("processing index = ", index, "count = ", count)
            index = str(index)
            pre_id = pre_id_dict[index]
            id = index_id_dict[index]
            data_id = index_dataid_dict[index]
            benchmark = dataid_benchmark_dict[data_id]
            vul_type = type_dict[index_vultype_dict[index]]

            source_files = os.listdir(os.path.join(self.config.sample_source_path, id))

            with open(os.path.join(self.config.sample_slice_path, index), 'r') as slice_f:
                slice_content = slice_f.readlines()

            with_line_slice_file_name = os.listdir(os.path.join(self.config.slice_with_line_path, vul_type, pre_id))[0]
            with open(os.path.join(self.config.slice_with_line_path, vul_type, pre_id, with_line_slice_file_name), 'r') as with_line_slice_f:
                with_line_slice_content = with_line_slice_f.read()

            with_line_slice_list = with_line_slice_content.split(end_line)
            if len(with_line_slice_list)<2:
                print(with_line_slice_content)
            map_with_line_slice = []
            approximate_with_line_slice = []
            search_str = data_id.split()[1] + ' ' + benchmark
            approximate_search_str = data_id.split()[1] + ' ' + ' '.join(benchmark.split()[:-1])
            for with_line_slice in with_line_slice_list:
                lines = with_line_slice.split('\n')
                if len(lines) > 2:
                    information = lines[0]
                    if search_str in information:
                        map_with_line_slice.append(with_line_slice)
                    if approximate_search_str in information:
                        approximate_with_line_slice.append(with_line_slice)
            # 处理 没匹配的切片
            line_dict = {}
            if map_with_line_slice == []:
                if approximate_with_line_slice == []:
                    for ori_line, content in enumerate(slice_content):
                        if content == '\n':
                            line_dict[ori_line] = [-1, None]
                            continue
                        find = False
                        for source_file in source_files:
                            if not (source_file.endswith('.c') or source_file.endswith('.cpp')):
                                continue
                            with open(os.path.join(self.config.sample_source_path, id, source_file), 'r') as sf:
                                source_content = sf.readlines()
                            for map_line_number, map_content in enumerate(source_content):
                                if ''.join(content.split()) in ''.join(map_content.strip().split()):
                                    line_dict[ori_line] = [map_line_number+1, source_file]
                                    find = True
                                    break
                            if find:
                                break
                        if not find:
                            line_dict[ori_line] = [-1, None]
                    index_line_dict[index] = line_dict
                    continue
                else:
                    m = float("inf")
                    for ap in approximate_with_line_slice:
                        benchmark_line_number = int(benchmark.split()[-1])
                        map_benchmark_line_number = int(ap[0].split()[-1])
                        if abs(benchmark_line_number - map_benchmark_line_number) < m:
                            map_with_line_slice = [ap]
                            m = benchmark_line_number - map_benchmark_line_number

            if len(map_with_line_slice) > 1:
                find = False
                for i, map_with_line_content in enumerate(map_with_line_slice):
                    map_content0 = "".join(map_with_line_content[1].split()[:-1])
                    map_content1 = "".join(map_with_line_content[2].split()[:-1])
                    content0 = "".join(slice_content[0].split())
                    content1 = "".join(slice_content[0].split())
                    if content0 == map_content0 and content1 == map_content1:
                        map_with_line_slice = [map_with_line_slice[i]]
                        find = True
                        break
                if not find:
                    map_with_line_slice = [map_with_line_slice[0]]

            map_with_line_slice_list = map_with_line_slice[0].split('\n')
            for ori_line, content in enumerate(slice_content):
                if content == '\n':
                    line_dict[ori_line] = [-1, None]
                    continue
                find = False
                for map_line in map_with_line_slice_list[1:]:
                    if len(map_line.split()) < 2:
                        continue
                    map_content = ''.join(map_line.split()[:-1])
                    map_line_number = map_line.split()[-1]
                    try:
                        map_line_number = int(map_line_number)
                    except:
                        continue
                    if ''.join(content.split()) == map_content:
                        map_file_name = None
                        for source_file in source_files:
                            # 查找该行所属的文件
                            if not(source_file.endswith('.c') or source_file.endswith('.cpp')):
                                continue
                            with open(os.path.join(self.config.sample_source_path, id, source_file), 'r') as sf:
                                source_content = sf.readlines()
                            if int(map_line_number) - 1 < len(source_content) and map_content in ''.join(source_content[int(map_line_number) - 1].strip().split()):
                                map_file_name = source_file
                                break

                        if map_file_name == '':
                            # 处理文本匹配了但是没有匹配到file_name的情况
                            pre_source_file = line_dict[ori_line-1][1]
                            pre_line_number = line_dict[ori_line-1][0]
                            if pre_source_file != '' and int(map_line_number) > int(pre_line_number):
                                map_file_name = pre_source_file
                        line_dict[ori_line] = [map_line_number, map_file_name]
                        find = True
                if not find:
                    # 处理没有匹配上的行
                    candidates = []
                    for source_file in source_files:
                        if not (source_file.endswith('.c') or source_file.endswith('.cpp')):
                            continue
                        with open(os.path.join(self.config.sample_source_path, id, source_file), 'r') as sf:
                            source_content = sf.readlines()
                        for ln, map_content in enumerate(source_content):
                            if ''.join(content.split()) in ''.join(map_content.strip().split()):
                                candidates.append([ln+1, source_file])
                    if candidates == []:
                        line_dict[ori_line] = [-1, None]
                    else:
                        if ori_line > 0:
                            find = False
                            for candidate in candidates:
                                if candidate[1] == line_dict[ori_line-1][1] and candidate[0] > line_dict[ori_line-1][0]:
                                    line_dict[ori_line] = candidate
                                    find = True
                                    break
                            if not find:
                                line_dict[ori_line] = candidates[0]
                        else:
                            line_dict[ori_line] = candidates[0]
            index_line_dict[index] = line_dict
        index_line_number_path = os.path.join(self.config.temp_path, "index_line_number.json")
        with open(index_line_number_path, 'w') as index_line_number_f:
            json.dump(index_line_dict, index_line_number_f)

def main():
    train_tag = False  # whether or not training model
    sample_gen_tag = False # whether or not generating samples
    create_benchmark_tag = False
    create_line_number_tag = False
    create_vul_line_number_tag = False
    # sample_ids_json = '../resources/Dataset/sample_ids.json'
    sample_ids_json = os.path.join(config.root_path, 'sample_ids.json')

    train_data = pd.read_pickle(config.data_path + 'train/blocks.pkl')
    test_data = pd.read_pickle(config.data_path + 'test/blocks.pkl')
    # load embedding
    word2vec = Word2Vec.load(config.embedding_path + "/node_w2v_60").wv
    config.embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    config.embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    config.embedding_dim = word2vec.vectors.shape[1]
    config.vocab_size = word2vec.vectors.shape[0] + 1
    config.vocab = word2vec.vocab

    entry = Run_Entry(config)
    # train target model
    if train_tag:
        entry.train_target_model(train_data, test_data)

    # generate samples
    if sample_gen_tag:
        programs = entry.sample_Test(test_data)

    # 生成data_id与benchmark的对应关系
    if create_benchmark_tag:
        entry.create_benchmark()

    if not os.path.isfile(sample_ids_json):
        print("Sample first!")
        return

    with open(sample_ids_json, 'r') as sample_ids_file:
        programs_indexes = json.load(sample_ids_file)

    # 生成切片行号与原文件行号键值对
    if create_line_number_tag:
        entry.create_line_number(programs_indexes)

    if create_vul_line_number_tag:
        entry.create_vul_line_number(programs_indexes)

    # r = entry.test_multiple_program(target_model, test_data)
    # print(r)

    # predicted for single program
    # attack = AttackTarget(config)
    # target_model = attack.load_trained_model('adv_VulDetectModel_0.7.pt')
    # target_model = attack.load_trained_model('adv_VulDetectModel.pt')

    # 查看最重要的句子样子
    # nope = NopeTool(config, test_data)
    # nope.creat_importance_statement()

    # 测试变异测试
    # mutation = Mutation(config, programs_indexes, target_model, model_name='adv_VulDetectModel_0.7.pt')
    # mutation.attack()

    # 测试代码混淆
    replace_limit_number = 1
    # obfuscation = Obfuscation(config, replace_limit_number, programs_indexes, model_name="adv_VulDetectModel_0.7.pt")
    obfuscation = Obfuscation(config, replace_limit_number, programs_indexes)
    # obfuscation.macro_raplace_attack()
    # obfuscation.replace_line_attack()
    obfuscation.add_dead_code_attack()
    # obfuscation.replace_const_attack()
    # obfuscation.merge_function_attack()
    # obfuscation.unroll_loop_attack()

    # 测试攻击方法组合
    # replace_limit_number = 15
    # combination = Combination(config, replace_limit_number, programs_indexes, model_name="adv_VulDetectModel_0.7.pt")
    # combination = Combination(config, replace_limit_number, programs_indexes, model_name="VulDetectModel.pt")
    # combination = Combination(config, replace_limit_number, programs_indexes, model_name="vul_detect_zigzag.pt")
    # combination.combination_attack(random_tag=True)

    # combination.only_one_sample()

    # 遗传算法
    # limits = [3,3]
    # limits = 15
    # genetic = Genetic(config, limits, programs_indexes)
    # genetic.attack()

if __name__ == '__main__':
    config = MyConf('../Config/config.cfg')
    main()

    # json 载入进来后都是字符串类型