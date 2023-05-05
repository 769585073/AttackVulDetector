import os
import csv

root_path = '../resources/Log'

def main(file_name):
    file = os.path.join(root_path, file_name)
    with open(file, 'r') as f_txt:
        content = f_txt.readlines()
    with open(os.path.join(root_path, file_name.split(',')[0]+".csv"), 'w', encoding='utf-8', newline='') as r_csv_f:
        csv_writer = csv.writer(r_csv_f)
        csv_writer.writerow(['epochs', 'loss_train', 'accuracy_train', 'precision_train', 'recall_train', 'f1_train', 'loss_test', 'accuracy_test', 'precision_test', 'recall_test', 'f1_test',])
        i = 0
        best = [-1,0,0,0,0]
        while i < len(content):
            loss_content = content[i].split(':')
            measure_content = content[i+1].split(':')
            epochs = i//2
            loss_train = float(loss_content[2].split(',')[0])
            loss_test = float(loss_content[3].split(',')[0])
            accuracy_train = float(measure_content[2].split(',')[0])
            precision_train = float(measure_content[3].split(',')[0])
            recall_train = float(measure_content[4].split(',')[0])
            f1_train = float(measure_content[5].split('],')[0])
            accuracy_test = float(measure_content[7].split(',')[0])
            precision_test = float(measure_content[8].split(',')[0])
            recall_test = float(measure_content[9].split(',')[0])
            f1_test = float(measure_content[10].split(']')[0])
            if f1_test >= best[-1]:
                best = [epochs, accuracy_test, precision_test, recall_test, f1_test]
            csv_writer.writerow([epochs, loss_train, accuracy_train, precision_train, recall_train, f1_train, loss_test, accuracy_test, precision_test, recall_test, f1_test])
            i+=2
        csv_writer.writerow(best)
    print(best)


if __name__ == "__main__":
    # file_name = 'recurrent_VulDetectModel_log.txt'
    # file_name = 'adv_VulDetectModel_8000_log_all.txt'
    # file_name = 'adv_VulDetectModel_7200_log.txt'
    # file_name = 'adv_VulDetectModel_6400_log.txt'
    # file_name = 'adv_VulDetectModel_5600_log.txt'

    # file_name = 'adv_VulDetectModel_1.0_log.txt'
    file_name = 'fine_tuning_VulDetectModel_0.7_log.txt'
    # file_name = 'only_success_adv_VulDetectModel_1.0_log.txt'
    main(file_name)
