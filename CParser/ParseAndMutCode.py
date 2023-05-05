# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/5/16 15:40
# @Function:
import json
import os
import random
import re
import string

from CParser.ParserVisitor import ParserVisitor
from CParser.cGrammer.CLexer import CLexer
from CParser.cGrammer.CParser import CParser
from antlr4 import FileStream, CommonTokenStream, InputStream


class ParseAndMutCode:
    def parse_statement(self, statement, file_name):
        inputs = InputStream(statement)
        lexer = CLexer(inputs)
        stream = CommonTokenStream(lexer)
        parser = CParser(stream)
        tree = parser.compilationUnit()
        mv = ParserVisitor()
        mv.visit(tree)
        # for x in mv.prolog_list.keys():
        #     prolog = mv.prolog_list[x]
        #     prolog.toString()
        # 常量替换
        # self.create_const_replace(mv.const_variable, file_name)

    def translate_c_replace_const(self, sample_source_path, temp_path):
        # 生成filename_outspace.json {id/filename：{ori_line： replaced_content}}
        # replaced_content两行 第一行声明 第二行替换后的代码
        # 错误不在PaserVisitor中的
        problem_ids = ['150237', '151726', '152563', '152113', 'CVE-2015-4511', '153160', 'CVE-2015-0824', '151542', '151618', '153168', '153202', 'CVE-2016-1957', 'CVE-2016-6213']
        count = 0
        filename_replace_const_dict = {}
        for id in os.listdir(sample_source_path):
            print(id, count)
            count += 1
            files = os.listdir(os.path.join(sample_source_path, id))
            if id in problem_ids:
                continue
            for file in files:
                if not(file.endswith('.c') or file.endswith('.cpp')):
                    continue
                file_name = os.path.join(sample_source_path, id, file)
                with open(file_name, 'r') as f:
                    code_content = f.readlines()
                result = ''
                for statement in code_content:
                    result += statement

                inputs = InputStream(result)
                lexer = CLexer(inputs)
                lexer.removeErrorListeners()
                stream = CommonTokenStream(lexer)
                parser = CParser(stream)
                parser.removeErrorListeners()
                tree = parser.compilationUnit()
                mv = ParserVisitor()
                mv.visit(tree)

                line_replace_dict = {}
                for origin_line in range(len(code_content)):
                    if origin_line+1 not in mv.const_int_dict.keys() and origin_line+1 not in mv.const_string_dict.keys():
                        continue
                    new_code = code_content[origin_line].split('/*')[0]
                    declaration_line1 = ''
                    if origin_line + 1 in mv.const_string_dict.keys() and mv.const_string_dict[origin_line+1] != []:
                        declaration_line1 = 'const char* '
                        for const in mv.const_string_dict[origin_line+1]:
                            new_const_name = ''.join(random.choice(string.ascii_uppercase) for i in range(5))
                            new_code = new_code.replace(const, new_const_name)
                            declaration_line1 = declaration_line1 + new_const_name + ' = ' + str(const) + ','
                        declaration_line1 = declaration_line1[:-1] + ';'

                    declaration_line = ''
                    if origin_line+1 in mv.const_int_dict.keys() and mv.const_int_dict[origin_line+1] != []:
                        declaration_line = 'const int '
                        for const in mv.const_int_dict[origin_line+1]:
                            new_const_name = ''.join(random.choice(string.ascii_uppercase) for i in range(5))
                            new_code = new_code.replace(const, new_const_name)
                            declaration_line = declaration_line + new_const_name + ' = ' + str(const) + ','
                        declaration_line = declaration_line[:-1] + ';'

                    replace_content = []
                    if declaration_line!= '':
                        replace_content.append(declaration_line.strip())
                    if declaration_line1!='':
                        replace_content.append(declaration_line1.strip())
                    replace_content.append(new_code.strip())
                    line_replace_dict[origin_line] = replace_content
                key = id + '/' + file
                filename_replace_const_dict[key] = line_replace_dict

        filename_replace_const_path = os.path.join(temp_path, 'filename_replace_const.json')
        with open(filename_replace_const_path, 'w') as filename_replace_const_jsn:
            json.dump(filename_replace_const_dict, filename_replace_const_jsn)

    def translate_c_merge_function(self, sample_source_path, temp_path):
        problem_ids = ['150237', '151726', '152563', '152113', 'CVE-2015-4511', '153160', 'CVE-2015-0824', '151542', '151618', '153168', '153202', 'CVE-2016-1957', 'CVE-2016-6213']
        count = 0
        filename_merge_function_dict = {}
        for id in os.listdir(sample_source_path):
            print(id, count)
            count += 1
            files = os.listdir(os.path.join(sample_source_path, id))
            if id in problem_ids:
                continue
            func_definition_dict = {}
            line_input_space = {}
            line_output_space = {}
            for file in files:
                if not(file.endswith('.c') or file.endswith('.cpp')):
                    continue
                file_name = os.path.join(sample_source_path, id, file)
                with open(file_name, 'r') as f:
                    code_content = f.readlines()
                result = ''
                for statement in code_content:
                    result += statement
                inputs = InputStream(result)
                lexer = CLexer(inputs)
                lexer.removeErrorListeners()
                stream = CommonTokenStream(lexer)
                parser = CParser(stream)
                parser.removeErrorListeners()
                tree = parser.compilationUnit()
                mv = ParserVisitor()
                mv.visit(tree)
                func_definition_dict[file] = mv.func_definition_dict
                line_input_space[file] = mv.line_input_space
                line_output_space[file] = mv.line_output_space

            for file in files:
                if not(file.endswith('.c') or file.endswith('.cpp')):
                    continue
                file_name = os.path.join(sample_source_path, id, file)
                with open(file_name, 'r') as f:
                    code_content = f.readlines()

                line_fucntion_dict = {}
                for origin_line in range(len(code_content)):
                    if origin_line+1 in line_input_space[file].keys():
                        # 输入空间有东西
                        for variable in line_input_space[file][origin_line+1]:
                            # 判断每一个变量
                            find = False
                            for defin_file in func_definition_dict.keys():
                                for func_definition_line in func_definition_dict[defin_file].keys():
                                    # 与每个文件中的函数定义名称相比较
                                    for func_name in func_definition_dict[defin_file][func_definition_line]:
                                        if func_name == variable:
                                            line_fucntion_dict[origin_line] = [func_definition_line-1, defin_file]
                                            find = True
                                            break
                                    if find:
                                        break
                                if find:
                                    break
                    if origin_line+1 in line_output_space[file].keys():
                        # 输出空间有东西
                        for variable in line_output_space[file][origin_line+1]:
                            # 判断每一个变量
                            find = False
                            for defin_file in func_definition_dict.keys():
                                for func_definition_line in func_definition_dict[defin_file].keys():
                                    # 与每个文件中的函数定义名称相比较
                                    for func_name in func_definition_dict[defin_file][func_definition_line]:
                                        if func_name == variable:
                                            line_fucntion_dict[origin_line] = [func_definition_line-1, defin_file]
                                            find = True
                                            break
                                    if find:
                                        break
                                if find:
                                    break
                key = id + '/' + file
                filename_merge_function_dict[key] = line_fucntion_dict
        filename_merge_function_path = os.path.join(temp_path, 'filename_merge_function.json')
        with open(filename_merge_function_path, 'w') as filename_merge_function_jsn:
            json.dump(filename_merge_function_dict, filename_merge_function_jsn)

    def translate_c(self, file_name):
        with open(file_name, 'r') as f:
            code_content=f.readlines()
        result = ''
        for statement in code_content:
            result += statement
        self.parse_statement(result, file_name)

    def parser_code_gadget(self,file_name):
        with open(file_name, 'r') as f:
            code_content = f.read()
        inputs = InputStream(code_content)
        lexer = CLexer(inputs)
        stream = CommonTokenStream(lexer)
        parser = CParser(stream)
        tree = parser.compilationUnit()
        mv = ParserVisitor()
        mv.visit(tree)
        for x in mv.prolog_list.keys():
            prolog = mv.prolog_list[x]
            prolog.toString()
        print(mv)

    # 交换语句or添加print语句的判定
    def parse_the_slice_line(self, slice_file_name, code_position, slice_line, print=False):
        with open(slice_file_name, 'r') as f:
            slice_contet = f.readlines()
        if not print:
            # 交换语句的文件判定
            code, code_line, down_code, down_line = '', -1, '', -1
            try:
                code, code_line = slice_contet.__getitem__(slice_line - 1).strip().rsplit(' ', 1)
                if slice_line< len(slice_contet):
                    down_code, down_line = (slice_contet.__getitem__(slice_line)).strip().rsplit(' ', 1)
            except ValueError:
                print("error")
            code_line = int(code_line)
            down_line = int(down_line)
            negative_keywords = ['for', 'while', 'dowhile', 'if', 'else', 'switch', 'case', 'default', 'continue', 'break', 'return']
            if any(s in code for s in negative_keywords) or any(s in down_code for s in negative_keywords):
                print("交换行包括控制语句，无法交换")
                return '', -1, -1
            if down_line != code_line + 1:
                print("行不相邻无法交换")
                return '', -1, -1
            for root, ds, fs in os.walk(code_position):
                for f in fs:
                    fullname = os.path.join(root, f)
                    if fullname == slice_file_name or fullname.__contains__('adversival'):
                        continue
                    with open(fullname, 'r') as f:
                        code_content = f.readlines()
                    if code_line < len(code_content) and down_line < len(code_content):
                        temp1 = re.sub('[\s;+]', '', code_content.__getitem__(code_line-1))
                        temp2 = re.sub('[\s;+]', '', code)
                        temp3 = re.sub('[\s;+]', '', code_content.__getitem__(down_line-1))
                        temp4 = re.sub('[\s;+]', '', down_code)
                        if temp1 == temp2 and temp3 == temp4:
                            return fullname, code_line, down_line
            return '', -1, -1
        else:
            try:
                code, code_line = slice_contet.__getitem__(slice_line - 1).strip().rsplit(' ', 1)
                code_line = int(code_line)
            except ValueError:
                print("error")
            # 添加print的文件判定
            for root, ds, fs in os.walk(code_position):
                for f in fs:
                    fullname = os.path.join(root, f)
                    if fullname == slice_file_name or fullname.__contains__('adversival'):
                        continue
                    with open(fullname, 'r') as f:
                        code_content = f.readlines()
                    if code_line < len(code_content):
                        temp1 = re.sub('[\s;+]', '', code_content.__getitem__(code_line-1))
                        temp2 = re.sub('[\s;+]', '', code)
                        if temp1 == temp2:
                            return fullname, code_line
            return '', -1

    def translate_c_exchange_line(self, sample_source_path, temp_path):
        # 生成filename_outspace.json {id/filename：{ori_line： down_line}}
        # 错误不在PaserVisitor中的
        problem_ids = ['150237', '151726', '152563', '152113', 'CVE-2015-4511', '153160', 'CVE-2015-0824', '151542', '151618', '153168', '153202', 'CVE-2016-1957', 'CVE-2016-6213']
        count = 0
        filename_exchange_line_dict = {}
        for id in os.listdir(sample_source_path):
            print(id, count)
            count += 1
            files = os.listdir(os.path.join(sample_source_path, id))
            if id in problem_ids:
                for file in files:
                    key = id + '/' + file
                    filename_exchange_line_dict[key] = {}
                continue
            for file in files:
                if not(file.endswith('.c') or file.endswith('.cpp')):
                    continue
                file_name = os.path.join(sample_source_path, id, file)

                with open(file_name, 'r') as f:
                    code_content = f.readlines()
                result = ''
                for statement in code_content:
                    result += statement

                inputs = InputStream(result)
                lexer = CLexer(inputs)
                lexer.removeErrorListeners()
                stream = CommonTokenStream(lexer)
                parser = CParser(stream)
                parser.removeErrorListeners()
                tree = parser.compilationUnit()
                mv = ParserVisitor()
                mv.visit(tree)
                exchange_line_dict = {}
                for origin_line in range(len(code_content)-1):
                    down_line = origin_line+1
                    origin_line_input_space = self.preprocess(mv.line_input_space.get(origin_line))
                    origin_line_output_space = self.preprocess(mv.line_output_space.get(origin_line))
                    down_line_input_space = self.preprocess(mv.line_input_space.get(down_line))
                    down_line_output_space = self.preprocess(mv.line_output_space.get(down_line))

                    if (set(origin_line_input_space) & set(down_line_output_space)) or (set(down_line_input_space) & set(origin_line_output_space)) or (
                            set(origin_line_output_space) & set(down_line_output_space)):
                        exchange_line_dict[origin_line] = -1
                    else:
                        exchange_line_dict[origin_line] = down_line
                key = id + '/' + file
                filename_exchange_line_dict[key] = exchange_line_dict

        filename_exchange_line_path = os.path.join(temp_path, 'filename_exchange_line.json')
        with open(filename_exchange_line_path, 'w') as filename_exchange_line_jsn:
            json.dump(filename_exchange_line_dict, filename_exchange_line_jsn)

    # 可以交换，进行交换
    # def translate_c_exchange_line(self, file_name, origin_line, down_line):
    #     # 重写优化时间
    #     with open(file_name, 'r') as f:
    #         code_content = f.readlines()
    #     result = ''
    #     for statement in code_content:
    #         result += statement
    #     return self.parse_statement_exchange_line(file_name, result, origin_line, down_line)
    #
    #
    # def parse_statement_exchange_line(self, file_name, statement, origin_line, down_line):
    #     inputs = InputStream(statement)
    #     lexer = CLexer(inputs)
    #     lexer.removeErrorListeners()
    #     stream = CommonTokenStream(lexer)
    #     parser = CParser(stream)
    #     parser.removeErrorListeners()
    #     tree = parser.compilationUnit()
    #     mv = ParserVisitor()
    #     mv.visit(tree)
    #     # for x in mv.prolog_list.keys():
    #     #     prolog = mv.prolog_list[x]
    #     #     if len(prolog.children_id) == 0 and (prolog.line == origin_line or prolog.line == down_line):
    #     #         prolog.toString()
    #     # print(mv.line_input_space)
    #     # print(mv.line_output_space)
    #     origin_line_input_space = self.preprocess(mv.line_input_space.get(origin_line))
    #     origin_line_output_space = self.preprocess(mv.line_output_space.get(origin_line))
    #     down_line_input_space = self.preprocess(mv.line_input_space.get(down_line))
    #     down_line_output_space = self.preprocess(mv.line_output_space.get(down_line))
    #
    #     # print("原始行%d的输入输出空间"%origin_line, origin_line_input_space, origin_line_output_space)
    #     # print("下一行%d的输入输出空间"%down_line, down_line_input_space, down_line_output_space)
    #     if (set(origin_line_input_space) & set(down_line_output_space)) or (set(down_line_input_space) & set(origin_line_output_space)) or (
    #             set(origin_line_output_space) & set(down_line_output_space)):
    #         # print("输入输出空间相互影响不能交换")
    #         return False
    #     else:
    #         return True
    #         # self.create_adversival_sample_exchange_line(file_name, origin_line, down_line)

    def preprocess(self, line_space):
        # 预处理空间，主要是解决数组声明(包含[]),函数调用(方法名和参数)在visitor中遍历不出来的问题,通过正则表达式匹配解决
        if line_space is not None:
            for space in line_space:
                if re.search(r'[].*&,(.*?)[]', space):
                    line_space.extend(one.strip() for one in re.split(r'[].*&,(.*?)[]', space) if one !='' and one != ' ')
        else:
            return []
        return line_space

    def create_adversival_sample_exchange_line(self, file_name, origin_line, down_line):
        new_file_name = file_name.rsplit('.', 1)[0] + '_adversival_' + str(origin_line) + '.' + file_name.rsplit('.', 1)[1]
        with open(file_name, 'r') as f:
            lines = f.readlines()
        fo = open(new_file_name, 'w')
        for j in range(len(lines)):
            if j == down_line-1:
                fo.write(lines[origin_line-1])
            elif j == origin_line-1:
                fo.write(lines[down_line-1])
            else:
                fo.write(lines[j])
        fo.close()

    def translate_c_add_print(self, sample_source_path, temp_path):
        # 生成filename_outspace.json {id/filename：{line_number： outspace}}
        # 错误不在PaserVisitor中的
        problem_ids = ['150237', '151726', '152563', '152113', 'CVE-2015-4511', '153160', 'CVE-2015-0824', '151542', '151618', '153168', '153202', 'CVE-2016-1957', 'CVE-2016-6213']
        count = 0
        filename_outspace_dict = {}
        for id in os.listdir(sample_source_path):
            print(id, count)
            count += 1
            # if count < 786:
            #     continue
            files = os.listdir(os.path.join(sample_source_path, id))
            if id in problem_ids:
                continue
            for file in files:
                if not(file.endswith('.c') or file.endswith('.cpp')):
                    continue
                file_name = os.path.join(sample_source_path, id, file)
                with open(file_name, 'r') as f:
                    code_content = f.readlines()
                result = ''
                for statement in code_content:
                    result += statement
                inputs = InputStream(result)
                lexer = CLexer(inputs)
                # 不要warning
                lexer.removeErrorListeners()
                stream = CommonTokenStream(lexer)
                parser = CParser(stream)
                # 不要warning
                parser.removeErrorListeners()
                tree = parser.compilationUnit()
                mv = ParserVisitor()
                mv.visit(tree)

                origin_line_outspace_dict = {}
                for origin_line in range(len(code_content)):
                    outspace = []
                    if origin_line+1 in mv.line_output_space.keys():
                        outspace = list(set(mv.line_output_space[origin_line+1]))
                    origin_line_outspace_dict[origin_line] = outspace
                key = id + '/' + file
                filename_outspace_dict[key] = origin_line_outspace_dict

        filename_outspace_path = os.path.join(temp_path, 'filename_outspace.json')
        with open(filename_outspace_path, 'w') as filename_outspace_jsn:
            json.dump(filename_outspace_dict, filename_outspace_jsn)


    # 可以添加print，进行添加
    # def translate_c_add_print(self, file_name, origin_line):
    # # 修改了一下 这个程序很慢
    #     with open(file_name, 'r') as f:
    #         code_content = f.readlines()
    #     result = ''
    #     for statement in code_content:
    #         result += statement
    #     inputs = InputStream(result)
    #     lexer = CLexer(inputs)
    #     # 不要warning
    #     lexer.removeErrorListeners()
    #     stream = CommonTokenStream(lexer)
    #     parser = CParser(stream)
    #     # 不要warning
    #     parser.removeErrorListeners()
    #     tree = parser.compilationUnit()
    #     mv = ParserVisitor()
    #     mv.visit(tree)
    #     # for x in mv.prolog_list.keys():
    #     #     prolog = mv.prolog_list[x]
    #     #     if len(prolog.children_id) == 0 and (prolog.line <= origin_line):
    #     #         prolog.toString()
    #
    #
    #     ret = set()
    #     # for i in range(origin_line, 0, -1):
    #         # if mv.line_output_space.get(i) is not None:
    #             # line_output_space = set(mv.line_output_space.get(i))
    #             # self.create_adversival_sample_print(file_name, origin_line, line_output_space)
    #             # break
    #     if origin_line+1 in mv.line_output_space.keys():
    #         ret = set(mv.line_output_space[origin_line+1])
    #
    #     return ret
    #     # if i==-1:
    #     #     print('无法生成带变量名的控制语句')


    def create_adversival_sample_print(self, file_name, origin_line, variable_declaration):
        with open(file_name, 'r') as f:
            lines = f.readlines()
        for i in range(len(variable_declaration)):
            new_file_name = file_name.rsplit('.', 1)[0] + '_adversival_' + str(origin_line) + '_' + str(i) + '.' + file_name.rsplit('.', 1)[1]
            fo = open(new_file_name, 'w')
            for j in range(len(lines)):
                if j == origin_line:
                    print_string = 'printf("%x\\n",' + '&'+list(variable_declaration).__getitem__(i) + ');\n'
                    fo.write(print_string)
                fo.write(lines[j])
            fo.close()

    # 创建常量替换
    def create_const_replace(self, const_variable, file_name):
        with open(file_name, 'r') as f:
            code_content = f.readlines()
        origin_code = ''
        new_code = ''
        for statement in code_content:
            origin_code += statement
        for value in const_variable.values():
            for one in value:
                new_const_name = ''.join(random.choice(string.ascii_uppercase) for i in range(5))
                # print(new_const_name)
                new_code = origin_code.replace(one, new_const_name)
                origin_code = new_code
        new_file_name = file_name.rsplit('.', 1)[0] + '_adversival_const_replace.' + \
                        file_name.rsplit('.', 1)[1]
        fo = open(new_file_name, 'w')
        fo.write(new_code)
        fo.close()
        f.close()

if __name__ == '__main__':
    pm = ParseAndMutCode()
    # path = os.getcwd()  # 获取当前路径
    # file_name = os.path.join(path, "./62736_example/test.c")
    # file_name = os.path.join(path, "./62736_example/CWE121_Stack_Based_Buffer_Overflow__CWE129_listen_socket_45.c")
    # pm.translate_c(file_name)

    # 源代码级别
    # 首先从给定的切片文件以及想要混淆的行中找到源代码文件的行，由于可能是涉及多个文件，需要获取源代码文件夹下的文件循环遍历然后确定源代码的行位置信息
    # slice_line = 4
    # slice_file_name = os.path.join(path, "./62736_example/62736 with line of code.txt")
    # code_position = os.path.join(path, "./62736_example/")
    # 交换语句
    # code_file_name, origin_line, down_line = pm.parse_the_slice_line(slice_file_name, code_position, slice_line)
    # if origin_line != -1 and down_line != -1:
    #     pm.translate_c_exchange_line(code_file_name, origin_line, down_line)

    # 添加print
    # 首先收集变异行前出现的变量，此处打印其地址便可以省去判断直接打印，添加格式为printf('%x', &变量名);
    # 为了尽量使变量靠近变异行且有效，从变异行往上开始寻找，但仍需二次验证，可能存在找到的变量跨函数
    # code_file_name, origin_line = pm.parse_the_slice_line(slice_file_name, code_position, slice_line, True)
    # if origin_line != -1:
    #     pm.translate_c_add_print(code_file_name, origin_line)

    # 添加常量
    # const_variable = pm.translate_c(file_name)
    # tmp_p = "../resources/Dataset/SARD+NVD/20/113113\CWE762_Mismatched_Memory_Management_Routines__delete_array_char_calloc_17.cpp"
    tmp_p= "../resources/Dataset/SARD+NVD/31/CVE-2016-2550/linux_kernel_4.3.5_CVE_2016_2550_net_unix_af_unix.c"
    print(pm.parser_code_gadget(tmp_p))
