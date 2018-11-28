import os
import time
import copy

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from Method.LoadData import LoadData
from Method.Normalize import Normalize
from Algorithm.FNN import FNN
from Algorithm.LabelNN import LabelNN
from MkGraph.AccuracyPlot import AccuracyPlot
from MkGraph.ErrorPlot import ErrorPlot
from MkGraph.ModelScatter import ModelScatter

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
fnn_epoch = 2
fnn_random_size = 100

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
    print('<---Train the FNN ' + nn + ' Successfully--->')
    print('<----------------------------------------------->')

    rel_path = 'Experiment/Graph/method2/Best_FNN_' + nn + '_error_trend.png'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    ErrorPlot.error_trend(
        'Best_FNN_' + str(nn) + '_error_trend', len(fnn_copy.error_list), fnn_copy.error_list, abs_path)

    rel_path = 'Experiment/Graph/method2/Accuracy vs FNN' + str(nn) + '.png'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    AccuracyPlot.build_accuracy_plot(
        'Accuracy vs FNN'+str(nn), np.array([i for i in range(1, len(all_nn_accuracy) + 1, 1)]),
        all_nn_accuracy, abs_path)

    return fnn_copy, accuracy, matrix

# FNN 對照索引值
# C1_0 1
# C1_1 2
# C2_0 3
# C2_1 4
# C2_2 5
# C3_0 6
# C3_1 7
# C4_0 8
# C4_1 9
# C4_2 10
# C4_3 11
# C5   12
# C6   13


def label_convert(data):
    result = np.array([])
    for label in data:
        if label == 'C1_0':
            result = np.append(result, 1)
        elif label == 'C1_1':
            result = np.append(result, 2)
        elif label == 'C2_0':
            result = np.append(result, 3)
        elif label == 'C2_1':
            result = np.append(result, 4)
        elif label == 'C2_2':
            result = np.append(result, 5)
        elif label == 'C3_0':
            result = np.append(result, 6)
        elif label == 'C3_1':
            result = np.append(result, 7)
        elif label == 'C4_0':
            result = np.append(result, 8)
        elif label == 'C4_1':
            result = np.append(result, 9)
        elif label == 'C4_2':
            result = np.append(result, 10)
        elif label == 'C4_3':
            result = np.append(result, 11)
        elif label == 'C5':
            result = np.append(result, 12)
        elif label == 'C6':
            result = np.append(result, 13)
        else:
            print('Error Label')
    return result


def test_model(fnn_model):
    org_data, org_label = LoadData.get_method2_test()
    X_train, X_test, y_train, y_test = train_test_split(org_data, org_label, test_size=0.3)
    print('<---Test Model Start--->')
    output_list = np.array([])
    for model in fnn_model:
        fnn = FNN(
            fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size,
            model.mean, model.stddev, model.weight, fnn_lr, 1)
        output = fnn.testing_model(X_test)
        output_list = np.append(output_list, output)

    y_label = label_convert(y_test)
    output_list = output_list.reshape(-1, len(fnn_model))
    output_list = Normalize.normalization(output_list)
    label_pred = LabelNN.label_encode(output_list)
    for x, y in zip(output_list, y_label):
        print(x, ' ', y)

    C_matrix = confusion_matrix(y_label, label_pred)
    C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)

    print('This is the confusion matrix(test_all_model)\n', C_matrix)
    # print(C_matrix)
    # print(C_accuracy)

    print('<---Test Model Successfully--->')
    print('<----------------------------------------------->')
    return C_accuracy


# def show_model(mean, stddev, weight):
#     data = np.array([])
#     output = np.array([])
#     for i in range(-10, 10, 1):
#         for j in range(-10, 10, 1):
#             for k in range(-10, 10, 1):
#                 tmp = np.append(data, np.array([i, j, k]))
#     data = data.reshape(1, -1, 3) / 10
#     fnn = FNN(
#         fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size,
#         mean, stddev, weight, fnn_lr, 1)
#
#     for element in data:
#         output = np.append(output, fnn.forward(element))
#     ModelScatter.output_scatter_3d(data, output, fnn_threshold)


if __name__ == '__main__':

    start = time.time()
    print('<---Train--->')

    all_label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    cluster_num = {'C1': 2, 'C2': 3, 'C3': 2, 'C4': 4, 'C5': 0, 'C6': 0}
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
        fnn_accuracy.append(accuracy)
        fnn_matrix.append(pd.DataFrame(matrix, columns=pd_header, index=pd_header))

    model_accuracy = test_model(fnn_model)
    print('Model Accuracy', model_accuracy)

    end = time.time()

    print('All cost time is (' + str(end-start) + ')')