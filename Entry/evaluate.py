import os
import csv
from Config.ConfigT import MyConf


split_token = '<EOL>'


def write_slice(statements, filename):
    with open(filename, 'w') as f:
        for statement in statements:
            if not statement.endswith('\n'):
                statement = statement+'\n'
            f.writelines(statement)


def get_statistics(config, file_name):
    skipped, pos_success, neg_success, total_query, pos, neg = 0, 0, 0, 0, 0, 0
    with open(os.path.join(config.result_path, file_name), 'r') as csv_f:
        reader = csv.reader(csv_f)
        content = list(reader)
        for line in content[1:1001]:
            index, ground_label, ori_label, ori_prob, ori_slice, adv_label, adv_prob, adv_slice, query_times = line
            if ground_label!=ori_label:
                skipped+=1
                continue
            if int(ori_label) == 0:
                neg+=1
            else:
                pos+=1
            if ori_label!=adv_label:
                if int(ori_label)==0:
                    neg_success+=1
                else:
                    pos_success+=1
            total_query += int(query_times)
            #     ori_slice = ori_slice.split(split_token)
            #     adv_slice = adv_slice.split(split_token)
            #     write_slice(ori_slice, os.path.join(config.temp_path, "adv_samples", index+".txt"))
            #     write_slice(adv_slice, os.path.join(config.temp_path, "adv_samples", index+"_adv.txt"))

        return skipped, pos_success, neg_success, pos, neg, total_query


def evaluate(config, file_name):
    skipped, pos_success, neg_success, pos, neg, total_query = get_statistics(config, file_name)
    pos_success_rate = pos_success / pos
    neg_success_rate = neg_success / neg
    total_success_rate = (pos_success + neg_success) / (pos + neg)
    avg_query = total_query/(pos + neg)

    print("skipped :", skipped)
    print("pos_success :", pos_success)
    print("neg_success :", neg_success)
    print("pos :", pos)
    print("neg :", neg)
    print("pos_success_rate :", pos_success_rate)
    print("neg_success_rate :", neg_success_rate)
    print("total_success_rate :", total_success_rate)
    print("avg_query : ", avg_query)


def main(config):
    # file_name = "genetic_VulDetectModel.pt(15)(all).csv"
    # file_name = "genetic_VulDetectModel.pt(15)(all)(90,20,0.6,0.6,better_init).csv"
    # file_name = "combination_VulDetectModel.pt(all,greedy,15).csv"
    # file_name = "combination_VulDetectModel.pt(all_with_unroll_loop,greedy,15).csv"
    # file_name = "combination_vul_detect_zigzag.pt(all,greedy,15).csv"
    file_name = "combination_VulDetectModel.pt(V+D,random,15).csv"
    evaluate(config, file_name)


def get_examples(config):
    file_name = "combination_VulDetectModel.pt(all,greedy,15).csv"
    skipped, pos_success, neg_success, total_query, pos, neg = 0, 0, 0, 0, 0, 0
    with open(os.path.join(config.result_path, file_name), 'r') as csv_f:
        reader = csv.reader(csv_f)
        content = list(reader)
        for line in content[1:1001]:
            index, ground_label, ori_label, ori_prob, ori_slice, adv_label, adv_prob, adv_slice, query_times = line
            if ground_label!=ori_label:
                skipped+=1
                continue
            if int(ori_label) == 0:
                neg+=1
            else:
                pos+=1
            if ori_label!=adv_label:
                if int(ori_label)==0:
                    neg_success+=1
                else:
                    pos_success+=1
            total_query += int(query_times)
            #     ori_slice = ori_slice.split(split_token)
            #     adv_slice = adv_slice.split(split_token)
            #     write_slice(ori_slice, os.path.join(config.temp_path, "adv_samples", index+".txt"))
            #     write_slice(adv_slice, os.path.join(config.temp_path, "adv_samples", index+"_adv.txt"))


if __name__ == '__main__':
    config = MyConf('../Config/config.cfg')
    # config = MyConf('../Config/config_defence.cfg')
    # main(config)