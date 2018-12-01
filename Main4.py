import os
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from Method.LoadData import LoadData
from Method.Normalize import Normalize
from Method.Export import Export
from Algorithm.FNN import FNN
from Algorithm.LabelNN import LabelNN
from MkGraph.AccuracyPlot import AccuracyPlot
from MkGraph.ErrorPlot import ErrorPlot
from MkGraph.ModelScatter import ModelScatter
from MkGraph.ConfusionMatrix import ConfusionMatrix

"""
# Fuzzy Neural Networks Structure
# Fuzzy Data Set are 6000
"""
fnn_label_size = 6
fnn_input_size = 3
fnn_membership_size = fnn_input_size * fnn_label_size
fnn_rule_size = 6
fnn_output_size = 1
fnn_lr = 0.001
fnn_epoch = 1
fnn_random_size = 1

fnn_threshold = 0.0

# dimension_reduce_algorithm = ['LLE', 'PCA', 'Isomap', 'NCA, 'tSNE']
dimension_reduce_algorithm = ['tSNE']


def train_fnn(nn):
    accuracy = 0.0
    matrix = np.array([])
    fnn_copy = FNN()
    all_nn_accuracy = np.array([])

    org_data, org_label = LoadData.get_method2_fnn_train(nn)
    org_label = np.array([1 if label == nn else 0 for label in org_label])
    X_train, X_test, y_train, y_test = train_test_split(org_data, org_label, test_size=0.3)
    # print(X_train, X_train.shape)
    # print(y_train, y_train.shape)

    print('<---Train the FNN ' + nn + ' Start--->')
    for i in range(fnn_random_size):
        # Random Generate the mean, standard deviation
        mean = np.array(
            [np.random.uniform(-1, 1) for _ in range(fnn_membership_size)])
        stddev = np.array(
            [np.random.uniform(0, 1) for _ in range(fnn_membership_size)])
        weight = np.array(
            [np.random.uniform(-1, 1) for _ in range(fnn_rule_size)])

        fnn = FNN(
            fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, mean, stddev, weight, fnn_lr, 1)
        fnn.training_model(fnn_epoch, X_train, y_train)

        test_output = fnn.testing_model(X_test)
        label_pred = [1 if values >= fnn_threshold else 0 for values in test_output]
        C_matrix = confusion_matrix(y_test, label_pred)
        C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)
        all_nn_accuracy = np.append(all_nn_accuracy, C_accuracy)
        # print(C_matrix)
        # print(C_accuracy)
        if C_accuracy > accuracy:
            fnn_copy = copy.deepcopy(fnn)
            accuracy = copy.deepcopy(C_accuracy)
            matrix = copy.deepcopy(C_matrix)
            print('swap')
    print('<---Train the FNN ' + nn + ' Successfully--->')
    print('<----------------------------------------------->')

    # rel_path = 'Experiment/Graph/method2/Best_FNN_' + nn + '_error_trend.png'
    # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    # ErrorPlot.error_trend(
    #     'Best_FNN_' + str(nn) + '_error_trend', len(fnn_copy.error_list), fnn_copy.error_list, abs_path)
    #
    # rel_path = 'Experiment/Graph/method2/Accuracy vs FNN' + str(nn) + '.png'
    # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    # AccuracyPlot.build_accuracy_plot(
    #     'Accuracy vs FNN'+str(nn), np.array([i for i in range(1, len(all_nn_accuracy) + 1, 1)]),
    #     all_nn_accuracy, abs_path)

    return fnn_copy, accuracy, matrix


# 產生hash table
def build_hash_table():
    dictionary = {'C1': 6, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5, 'C6': 4}
    index = 0
    hash_table = {}
    for key, value in dictionary.items():
        if value == 0:
           hash_table[key] = index
        for i in range(value):
            hash_table[key+'_'+str(i)] = index
            index += 1
    print('hash_table', hash_table)
    return hash_table

#
# def label_convert(data, hash_table):
#     result = np.array([])
#     for label in data:
#         result = np.append(result, hash_table[label])
#     return result


# Get a list of keys from dictionary which has the given value
def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return listOfKeys[0]


#######
def label_encoding(data, hash_table):
    result = np.array([])

    table = np.array([0 for _ in range(6)])
    for key, value in hash_table.items():
        table[int(key[1:2])-1] += 1
    # print('table', table)

    # count1 紀錄"只有一個數值大於threshold"的次數
    # count2 紀錄"有多個數值大於threshold"的次數
    count = [0 for _ in range(2)]
    for array in data:
        tmp = []

        # 找誰大於0
        for i in range(len(array)):
            if array[i] > fnn_threshold:
                tmp.append(i)
        # 錯的
        if len(tmp) == 0:
            idx = 2

        # 判斷是否只有1個或多個
        elif len(tmp) == 1:
            # print('只有一個數值大於threshold，為: ', tmp)
            key = getKeysByValue(hash_table, tmp[0])
            idx = int(key[1:2])
            count[0] += 1
            # print('idx-1', idx)

        else:
            # print('有多個數值大於threshold，為: ', tmp)
            # 看細分類投票比率
            record = np.array([0 for _ in range(6)])
            for e in tmp:
                key = getKeysByValue(hash_table, e)
                tt = int(key[1:2])
                record[tt-1] += 1
            vector = record/table
            idx = vector.argmax()+1
            count[1] += 1
            print('record', record)
            print('vector', vector)
            # print('idx-2', idx)

        result = np.append(result, idx)
        # print('count', count)

    return result, count


def test_model(fnn_model):
    org_data, org_label = LoadData.get_method2_test()
    X_train, X_test, y_train, y_test = train_test_split(org_data, org_label, test_size=0.3)

    # Convert y_test(28 category to 6 category)
    y_test = np.array([int(e[1:2]) for e in y_test])

    print('<---Test Model Start--->')
    output_list = np.array([])
    for model in fnn_model:
        fnn = FNN(
            fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size,
            model.mean, model.stddev, model.weight, fnn_lr, 1)
        output = fnn.testing_model(X_test)
        output_list = np.append(output_list, output)

    # y_label = label_convert(y_test, build_hash_table())
    output_list = output_list.reshape(-1, len(fnn_model))

    # 不再做正規畫了試試
    output_list = Normalize.normalization(output_list)

    label_pred, count = label_encoding(output_list, build_hash_table())
    # cnt = 0
    # for x, y in zip(output_list[0:10], y_test[0:10]):
    #     print(x, ' ', y, ' ', label_pred[cnt])
    #     cnt += 1

    for x, y in zip(y_test, label_pred):
        print('correct', x, '<->', 'predict', y)

    cnf_matrix = confusion_matrix(y_test, label_pred)
    # 做confusion matrix 的圖
    # plt.figure()
    # ConfusionMatrix.plot_confusion_matrix(cnf_matrix, classes=list(set(y_test)),
    #                       title='Confusion matrix(Final FNN Model)')

    cnf_accuracy = np.sum(cnf_matrix.diagonal()) / np.sum(cnf_matrix)

    print('FinalModel_Accuracy: ', accuracy_score(y_test, label_pred))

    print('This is the confusion matrix(test_all_model)\n', cnf_matrix)
    # print(C_matrix)
    # print(C_accuracy)

    print('<---Test Model Successfully--->')
    print('<----------------------------------------------->')
    return cnf_accuracy, count


if __name__ == '__main__':

    start = time.time()
    print('<---Train--->')

    all_label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    # cluster_num = {'C1': 6, 'C2': 7, 'C3': 7, 'C4': 4, 'C5': 3, 'C6': 0}
    # cluster_num = {'C1': 6, 'C2': 5, 'C3': 4, 'C4': 4, 'C5': 5, 'C6': 3}
    cluster_num = {'C1': 6, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5, 'C6': 4}
    nn_category = np.array([])
    for element in all_label:
        if cluster_num[element] == 0:
            nn_category = np.append(nn_category, element)
        else:
            for num in range(cluster_num[element]):
                nn_category = np.append(nn_category, element + '_' + str(num))
    print('nn_category', nn_category)

    # Store those values to describe the best model in fnn local training
    fnn_model, fnn_accuracy, fnn_matrix = ([] for _ in range(3))

    pd_header = ['T', 'F']
    for nn in nn_category:
        fnn, accuracy, matrix = train_fnn(nn)
        fnn_model.append(fnn)

        # 儲存 fnn model as .json
        rel_path = 'Experiment/Model/FNN/'
        abs_path = os.path.join(os.path.dirname(__file__), rel_path)
        Export.save_fnn_weight(nn, fnn, abs_path)

        fnn_accuracy.append(accuracy)
        fnn_matrix.append(pd.DataFrame(matrix, columns=pd_header, index=pd_header))

    print('fnn_accuracy', fnn_accuracy)
    for x, y in zip(nn_category, fnn_matrix):
        print('<----------->')
        print(x, '\n', y)

    model_accuracy, count = test_model(fnn_model)
    print('Model Accuracy', model_accuracy)
    print('count:', count)

    # 做出統計圖
    x_axis = ['Only One Choose', 'Not Only One Choose']
    plt.bar(x_axis, list(count))
    plt.show()

    end = time.time()

    print('All cost time is (' + str(end-start) + ')')
