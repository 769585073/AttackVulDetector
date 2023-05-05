import copy
import os
import json
import csv

from Target_model.RunNet import RunNet
from Utils.mapping import mapping
from Utils.get_tokens import create_tokens
from collections import OrderedDict
from Utils.Util import concat_statement

class Mutation:
    def __init__(self, config, sample_ids, target_model, model_name):
        # index就是sample的index
        self.config = config
        self.sample_ids = sample_ids
        self.runnet = RunNet(config)
        self.target_model = target_model
        self.check_model_num = 0
        self.pre_id_dict = None
        self.ids_dict = None
        self.funcname_dict = None
        self.index_label_dict = None
        self.index_line_number = None
        self.index_vul_line_number_dict = None
        self.model_name = model_name

    def create_mutation_location_dict(self):
        # 生成并保存mutation_location_dict， 形式是{id/filename:{line:[[mutation_content, mutator]...]}}
        mutation_path = self.config.mutation_path
        ids = os.listdir(mutation_path)
        count = 0
        mutation_location_dict = {}
        for id in ids:
            count += 1
            print(id, count)
            if not os.path.exists(os.path.join(mutation_path, id, "mutator.txt")):
                continue
            files = os.listdir(os.path.join(mutation_path, id))
            with open(os.path.join(mutation_path, id, "mutator.txt"), "r") as mu:
                mutators = mu.readlines()

            mut_dict = {}
            index = 0
            while index < len(mutators):
                mut_location = mutators[index][:-1]
                mut_dict[mut_location] = [mutators[index + 1][:-1], mutators[index + 5][:-1]]
                index += 7

            for file in files:
                if file.startswith('milu_output') or not(file.endswith('.c') or file.endswith('.cpp')):
                    continue
                line_mutation_dict = {}
                file_path = os.path.join(mutation_path, id, file)
                with open(file_path, 'r') as f:
                    file_content = f.readlines()
                n = len(file_content)
                for mut_location in mut_dict.keys():
                    mut_filename = mut_location.split('/')[0][12:]
                    if mut_filename!=file:
                        continue
                    mut_line = int(mut_dict[mut_location][1])
                    if mut_line<=0 or mut_line>=n:
                        continue
                    mut_file_path = os.path.join(mutation_path, id, mut_location, file)
                    with open(mut_file_path, 'r') as mut_f:
                        mut_file = mut_f.readlines()
                    if mut_line >= len(mut_file):
                        continue
                    mut_content = mut_file[mut_line-1].strip()
                    mut_mutator = mut_dict[mut_location][0]
                    if mut_line in line_mutation_dict.keys():
                        line_mutation_dict[mut_line].append([mut_content, mut_mutator])
                    else:
                        line_mutation_dict[mut_line] = [[mut_content, mut_mutator]]
                key = id + '/' + file
                mutation_location_dict[key] = line_mutation_dict
        mutation_location_dict_path = os.path.join(self.config.temp_path, 'mutation_location_dict.json')
        with open(mutation_location_dict_path, 'w') as mutation_location_dict_jsn:
            json.dump(mutation_location_dict, mutation_location_dict_jsn)

    def get_all_mutation(self, index, mutation_location_dict):
        # 获取所有变异体
        id = self.ids_dict[index]
        line_number_dict = self.index_line_number[index]
        with open(os.path.join(self.config.sample_slice_path, index), 'r') as slice_program_f:
            slice_program = slice_program_f.readlines()
        vul_line_list = self.index_vul_line_number_dict[index]
        mutations = []
        tmp = {}
        for line, content in enumerate(slice_program):
            if str(line) in vul_line_list:
                continue
            ori_line = str(line_number_dict[str(line)][0])
            file_name = line_number_dict[str(line)][1]
            if ori_line == "-1" or file_name == None:
                # 没找到行号和文件名称的行不进行变异
                continue
            key = id + '/' + file_name
            if key not in mutation_location_dict.keys():
                continue
            line_mutation_dict = mutation_location_dict[key]
            if ori_line not in line_mutation_dict.keys():
                continue
            mut_contents = line_mutation_dict[ori_line]
            for mut_content in mut_contents:
                mutation = slice_program[:line] + [mut_content[0]] + slice_program[line+1:]
                mutations.append(mutation)
                if line in tmp.keys():
                    tmp[line].append(mut_content[0])
                else:
                    tmp[line] = [mut_content[0]]
        return mutations, tmp

    def is_code_matched(self, original_code, slice_code):
        if len(original_code) != len(slice_code):
            return False
        for i, x in enumerate(original_code):
            if "".join(x.split()) != "".join(slice_code[i].split()[:-1]):
                return False
            # if x not in slice_code[i]:
            #     return False
        return True

    def get_slice(self, index, pre_id, slice_path):
        type_dict = {'Array usage': 'arrayslice', 'Pointer usage': 'pointslice', 'API function call': 'apislice',
                     'Arithmetic expression': 'integeslice'}
        end_line = '------------------------------'
        types = []

        # 这么写是因为program里面的漏洞类型与切片的漏洞类型对应不上
        for key in type_dict.keys():
            types.append(key)
        file_name = self.funcname_dict[index]

        ori_code_path = os.path.join(self.config.sample_slice_path, index)
        with open(ori_code_path, 'r') as ori_file:
            original_code = ori_file.read().splitlines()
        id = self.ids_dict[index]
        for type in types:
            files = os.listdir(os.path.join(slice_path, type_dict[type], pre_id))
            if files == None:
                continue
            for file in files:
                with open(os.path.join(slice_path, type_dict[type], pre_id, file), "r") as slice_file:
                    contents = slice_file.readlines()
                start = 0
                for i, line in enumerate(contents):
                    if end_line in line:
                        tmp_path = contents[start].split(' ')[1]
                        tmp_path_split = tmp_path.split('/')
                        if tmp_path_split[-1] == file_name and tmp_path_split[-2] == id and tmp_path_split[
                            -3] == pre_id and self.is_code_matched(original_code, contents[start + 1:i]):
                            return contents[start+1:i]
                        start = i + 1
        return None

    def get_mutation(self, mutation_path, program_id):
        # 返回所有等价变异体 形式如下
        # equivalent : [[mutation_path, [mutator, mutate_line]], ...]
        files = os.listdir(os.path.join(mutation_path, program_id))
        if "equivalent1.txt" in files:
            with open(os.path.join(mutation_path, program_id, "equivalent1.txt"), "r") as eq:
                mutations = eq.readlines()
            with open(os.path.join(mutation_path, program_id, "mutator.txt"), "r") as mu:
                mutators = mu.readlines()
            equivalent = []
            mut_dict = {}
            index = 0
            while index < len(mutators):
                mut_location = mutators[index][:-5]
                mut_dict[mut_location] = [mutators[index + 1][:-1], mutators[index + 5][:-1]]
                index += 7

            for equivalent_location in mutations:
                curr_file_name = equivalent_location.split('/')[0][12:]
                with open(os.path.join(mutation_path, program_id, curr_file_name), 'r') as orig_f:
                    code_lines = orig_f.readlines()
                lines = len(code_lines)
                if int(mut_dict[equivalent_location[:-1]][1]) < lines:
                    equivalent.append([equivalent_location[:-1], mut_dict[equivalent_location[:-1]]])
            return equivalent
        return None

    def get_single_mutations(self, index):
        # 为每一个程序返回它的变异体
        mutation_path = self.config.mutation_path
        slice_path = self.config.slice_with_line_path
        id = self.ids_dict[index]
        file_name = self.funcname_dict[index]
        pre_id = self.pre_id_dict[index]

        orginal_slice = self.get_slice(index, pre_id, slice_path)
        if orginal_slice == None:
            print("%s :pre_id = %s Program_id = %s file_name = %s is not in sardslice" % (index, pre_id, id, self.funcname_dict[index]))
            return None

        mutations = self.get_mutation(mutation_path, id)
        if mutations == []:
            print("No equivalent mutation created for %s!" % id)
            return None
        if mutations == None:
            print("Fail to create equivalent mutation for %s!" % id)
            return None

        slice_line_code_dict = OrderedDict()
        slice_code_line_dict = {}
        for code in orginal_slice:
            line = code[:-1].split(" ")[-1]
            slice_line_code_dict[line] = " ".join(code.split()[:-1])
            slice_code_line_dict[code] = line
        # print(slice_line_code_dict)

        mutation_slice = []
        for mutation in mutations:
            # print(mutation_path + id + "/" + mutation[0] + "/src/" + file_name)
            # if mutation[0].split('/')[0][12:] != file_name:
            #     continue
            with open(os.path.join(mutation_path, id, mutation[0], "src", file_name), "r") as mu:
                mutation_code = mu.readlines()
            mutation_line = mutation[1][1]

            if mutation_line in slice_line_code_dict.keys():
                mutation_slice_content = []
                for code in orginal_slice:
                    if slice_code_line_dict[code] == mutation_line and int(mutation_line) - 1 < len(mutation_code):
                        # print(len(mutation_code), mutation_line)
                        # print(mutation_code)
                        # print(mutation_path + id + "/" + mutation[0] + "/src/" + file_name)
                        mutation_slice_content.append(mutation_code[int(mutation_line) - 1].strip())
                    else:
                        mutation_slice_content.append(slice_line_code_dict[slice_code_line_dict[code]])
                mutation_slice.append(mutation_slice_content)
        return mutation_slice

    def test_mutation(self, detect_model, mutation):
        self.check_model_num += 1
        predicted_label, probability = self.runnet.predict_single(detect_model, mutation)
        return predicted_label, probability

    def norm(self, mutation):
        inst_statements = []
        for line in mutation:
            tokens = create_tokens(line)
            inst_statements.append(tokens)
        map_program, _ = mapping(inst_statements)
        return map_program

    def attack(self):
        # 测试所有变异体
        if not os.path.exists(os.path.join(self.config.temp_path, 'mutation_location_dict.json')):
            print("creating mutation_location_dict.json!")
            self.create_mutation_location_dict()

        with open(os.path.join(self.config.temp_path, 'mutation_location_dict.json'), 'r') as mutation_location_dict_jsn:
            mutation_location_dict = json.load(mutation_location_dict_jsn)

        with open(os.path.join(self.config.temp_path, 'index_vul_line_number_dict.json'), 'r') as index_vul_line_jsn:
            self.index_vul_line_number_dict = json.load(index_vul_line_jsn)
            # print(len(self.index_vul_line_number_dict))

        index_preid_path = os.path.join(self.config.temp_path, 'index_pre_id.json')
        with open(index_preid_path, 'r') as pid_jsn:
            self.pre_id_dict = json.load(pid_jsn)

        ids_path = os.path.join(self.config.temp_path, 'index_id.json')
        with open(ids_path, 'r') as id_jsn:
            self.ids_dict = json.load(id_jsn)

        funcname_path = os.path.join(self.config.temp_path, 'index_funcname.json')
        with open(funcname_path, 'r') as func_jsn:
            self.funcname_dict = json.load(func_jsn)

        index_label_path = os.path.join(self.config.temp_path, 'index_label.json')
        with open(index_label_path, 'r') as index_label_jsn:
            self.index_label_dict = json.load(index_label_jsn)

        index_line_number_path = os.path.join(self.config.temp_path, 'index_line_number.json')
        with open(index_line_number_path, 'r') as index_line_number_jsn:
            self.index_line_number = json.load(index_line_number_jsn)

        count = 0
        sucess = 0
        result_path = os.path.join(self.config.result_path, 'mutation_attack_'+self.model_name+'_results.csv')
        with open(result_path, 'w', encoding='utf-8', newline='') as r_csv_f:
            csv_writer = csv.writer(r_csv_f)
            csv_writer.writerow(['index', 'true_label', 'ori_label', 'ori_prob', 'ori_code', 'adv_label', 'adv_prob', 'adv_code', 'query_times'])
            index_line_mutation_dict = {}
            for index in self.sample_ids:
                self.check_model_num = 0
                index = str(index)
                # mutations = self.get_single_mutations(index) # 只使用等价变异体
                mutations, line_mutation_dict = self.get_all_mutation(index, mutation_location_dict)# 使用全部变异体
                index_line_mutation_dict[index] = line_mutation_dict# 保存可用变异信息
                # with open(os.path.join(self.config.sample_path, index), 'r', encoding='utf-8') as map_file:
                #     map_code = map_file.readlines()

                slice_program_path = os.path.join(self.config.sample_slice_path, index)
                with open(slice_program_path, 'r') as slice_program_f:
                    slice_program = slice_program_f.readlines()
                map_code = self.norm(slice_program)
                original_label, original_probability = self.test_mutation(self.target_model, map_code)
                best_score = original_probability
                adv_label = original_label
                adv_code = slice_program
                if mutations == None:
                    continue
                if original_label != int(self.index_label_dict[index]):
                    slice_program = concat_statement(slice_program)
                    csv_writer.writerow([index, self.index_label_dict[index], original_label, original_probability, slice_program, original_label, original_probability, slice_program, self.check_model_num])
                    continue
                else:
                    count += 1
                    for mutation in mutations:
                        # print(mutation)
                        map_program = self.norm(mutation)
                        if map_program != []:
                            mutation_label, mutation_probability = self.test_mutation(self.target_model, map_program)
                            if original_label != mutation_label:
                                # print(mutation)
                                # print(original_label, mutation_label)
                                sucess += 1
                                adv_label = mutation_label
                                adv_code = copy.deepcopy(mutation)
                                best_score = 1.0 - mutation_probability
                                break
                            else:
                                if mutation_probability < best_score:
                                    adv_code = copy.deepcopy(mutation)
                                    best_score = mutation_probability
                    print(index, count, sucess, sucess/count)
                    adv_code = concat_statement(adv_code)
                    slice_program = concat_statement(slice_program)
                    csv_writer.writerow([index, self.index_label_dict[index], original_label, original_probability, slice_program, adv_label, best_score, adv_code, self.check_model_num])
            csv_writer.writerow([sucess, count, sucess/count])

        if not os.path.exists(os.path.join(self.config.temp_path, "index_line_mutation_dict.json")):
            with open(os.path.join(self.config.temp_path, "index_line_mutation_dict.json"), 'w') as index_line_mutation_dict_jsn:
                json.dump(index_line_mutation_dict, index_line_mutation_dict_jsn)