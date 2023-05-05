import os
import torch
import torch.nn.functional as F
import numpy as np

from Attack.Obfuscation import Obfuscation


class NopeTool():
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.endline = '--------\n'
        self.obfuscation = Obfuscation(self.config, 0, None)
        self.ret = None

    def run_model_by_batch(self, dataloader, ori_label, model=None):
        if model == None:
            model = self.obfuscation.detect_model
        predicted = []
        probability = []
        for batch in dataloader:
            model.batch_size = len(batch)
            logits = model(batch)
            softs = F.softmax(logits, dim=1)
            preds = torch.max(softs, 1)[1]
            if ori_label == 0:
                tmp = torch.zeros_like(preds)
            else:
                tmp = torch.ones_like(preds)
            probs = torch.index_select(softs, 1, tmp).t()[0]
            predicted.extend(preds.cpu().data.numpy().tolist())
            predicted.extend(preds.cpu().data.numpy().tolist())
            probability.extend(probs.cpu().data.numpy().tolist())
        return predicted, probability

    def statement_impact(self, slice, ori_label):
        tmp_slice_list = []
        for l in range(len(slice)):
            tmp_slice = slice[:l] + slice[l+1:]
            tmp_slice_list.append(tmp_slice)
        dataloader = self.obfuscation.make_batch(tmp_slice_list)
        _, probs = self.run_model_by_batch(dataloader, ori_label)
        state_score_arr = np.array(probs)
        sort_index = np.argsort(state_score_arr)
        return sort_index[0], probs[sort_index[0]]

    def creat_importance_statement(self):
        with open(os.path.join(self.config.temp_path, 'importance_statement_content.txt'), 'w') as f:
            for index, program in self.data.iterrows():
                slice = program['orig_code']
                map = self.obfuscation.norm_program(slice)
                ori_label, ori_prob = self.obfuscation.predict_adv_program(map)
                best_statement_loc, prob = self.statement_impact(map, ori_label)
                f.writelines(str(index) + ' ' + str(ori_label) + ' ' + str(ori_prob) + '\n')
                for i, s in enumerate(slice):
                    context = str(i) + ' ' + s + '\n'
                    f.writelines(context)
                f.writelines(str(best_statement_loc) + ' ' + slice[best_statement_loc] + ' ' + str(prob) + '\n' + self.endline)
                print(index)
        print("done!!!")

    def get_candidate_stament(self, ori_label):
        # 获取影响最大的一个statement
        if self.ret != None:
            return self.ret[1-ori_label]
        self.ret = {}
        if not os.path.exists(os.path.join(self.config.temp_path, 'importance_statement_content.txt')):
            self.creat_importance_statement()
        with open(os.path.join(self.config.temp_path, 'importance_statement_content.txt'), 'r') as f:
            contents = f.read().split(self.endline)
        candidates_dict = {0:{}, 1:{}}
        for i, content in enumerate(contents):
            content = content.split('\n')
            if len(content) < 3:
                continue
            label = int(content[0].split(' ')[1])
            line = " ".join(content[-2].split(' ')[1:-1])
            score = float(content[0].split(' ')[-1]) - float(content[-2].split(' ')[-1])
            if line in candidates_dict[label].keys():
                candidates_dict[label][line] = max(score, candidates_dict[label][line])
            else:
                candidates_dict[label][line] = score
        for label in candidates_dict.keys():
            tmp = []
            for key in candidates_dict[label].keys():
                tmp.append([key, candidates_dict[label][key]])
            tmp.sort(key = lambda x : x[1], reverse=True)
            ret = []
            [ret.append([x[0]]) for x in tmp[:10]]
            self.ret[label] = ret
        return self.ret[1-ori_label]

