"""
Method 3, FNN + DNN(Keras)
"""


import os
import sys
import time
import copy
import datetime
import numpy as np
import pandas as pd


from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn import preprocessing

from Method.LoadData import LoadData
from Method.Export import Export
from Method.Normalize import Normalize
from Method.ReducedAlgorithm import ReducedAlgorithm as ra
from Algorithm.FNN import FNN
from Algorithm.LabelNN import LabelNN
from MkGraph.AccuracyPlot import AccuracyPlot
from MkGraph.ErrorPlot import ErrorPlot
from MkGraph.ModelScatter import ModelScatter
from MkGraph.ConfusionMatrix import ConfusionMatrix

"""
# Fuzzy Neural Networks Structure
# Fuzzy Data Set are 1000
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

fnn_threshold1 = 0.0
fnn_threshold2 = 0.0
fnn_threshold3 = 0.0
fnn_threshold4 = 0.0
fnn_threshold5 = 0.0
fnn_threshold6 = 0.0

"""
Dimension reduce algorithm
"""

dimension_reduce_algorithm = ['tSNE']


"""
Use the method to train FNN1 ~ FNN6
Need to record the mean, stddev, weight, 使用的降維演算法
para1 -> which fnn
para2 -> which algorithm is used to reduce algorithm for fnn
"""


def train_local_fnn(nn, algorithm):

    # Declare variables
    nn_mean, nn_stddev, nn_weight = (0.0 for _ in range(3))
    accuracy = 0.0
    matrix = np.array([])
    fnn_copy = FNN()
    loss_list = np.array([])

    # This variable is used to store the all accuracy
    all_nn_accuracy = np.array([])

    # Load file FNN_Train_data_' + str(num) + '.xlsx
    org_data, org_label = LoadData.get_method1_fnn_train(nn)
    org_label = np.array([1 if element == nn else 0 for element in org_label])
    # Reduce dimension and generate train/test data
    # reduced_data = ra.tsne(org_data, fnn_input_size)
    reduced_data = ra.pca(org_data, fnn_input_size)
    # Normalized
    normalized_data = Normalize.normalization(reduced_data)

    X_train, X_test, y_train, y_test = train_test_split(
        normalized_data, org_label, test_size=0.3)
    # print(X_train, X_train.shape)
    # print(y_train, y_train.shape)

    # Train the FNN
    print('<---Train the FNN' + str(nn) + ' Start--->')
    for i in range(fnn_random_size):
        # Random Generate the mean, standard deviation
        mean = np.array(
            [np.random.uniform(-1, 1) for _ in range(fnn_membership_size)])
        stddev = np.array(
            [np.random.uniform(0, 1) for _ in range(fnn_membership_size)])
        weight = np.array(
            [np.random.uniform(-1, 1) for _ in range(fnn_rule_size)])
        """
        # Generate FNN object to train
        # para1 -> fnn input layer size
        # para2 -> fnn membership layer size
        # para3 -> fnn rule layer size
        # para4 -> fnn output layer size
        # para5 -> random mean values
        # para6 -> random stddev values
        # para7 -> random weight values
        # para8 -> nn label type
        """
        fnn = FNN(
            fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, mean, stddev, weight, fnn_lr, 1)
        fnn.training_model(fnn_epoch, X_train, y_train)

        # Test the FNN model, save the one that has the best accuracy
        test_output = fnn.testing_model(X_test)
        label_pred = np.array(
            [1 if value > fnn_threshold else 0 for value in test_output])
        # print(y_test.shape)
        # print(label_pred.shape)
        # print(y_test)
        # print(label_pred)

        C_matrix = confusion_matrix(y_test, label_pred)
        C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)
        all_nn_accuracy = np.append(all_nn_accuracy, C_accuracy)
        if C_accuracy > accuracy:
            accuracy = copy.deepcopy(C_accuracy)
            fnn_copy = copy.deepcopy(fnn)
            matrix = copy.deepcopy(C_matrix)
            loss_list = copy.deepcopy(fnn.loss_list)

    print('<---Train the FNN' + str(nn) + ' Successfully--->')
    print('<----------------------------------------------->')

    # Choose the best FNN to Plot error trend
    print(os.getcwd())
    abs_path = '/Users/yurenchen/PycharmProjects/Rider-Behavior/Experiment/method3/Grpah/Best_FNN_'+str(nn)+'_error_trend.png'
    # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    print(abs_path)
    ErrorPlot.error_trend(
        'Best_FNN_'+str(nn)+'_error_trend',
         len(fnn_copy.error_list), fnn_copy.error_list, abs_path)

    # Choose the best Accuracy to Plot
    print(os.getcwd())
    abs_path = '/Users/yurenchen/PycharmProjects/Rider-Behavior/Experiment/metdod3/Grpah/Accuracy vs FNN' + str(nn) + '.png'
    # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    # print('AccuracyPlot', abs_path)
    AccuracyPlot.build_accuracy_plot(
        'Accuracy vs FNN' +
        str(nn), np.array([i for i in range(1, len(all_nn_accuracy) + 1, 1)]),
        all_nn_accuracy, abs_path)

    return fnn_copy, accuracy, matrix


"""
Test all model
"""


def test_all_model(fnn_attribute, lnn_attribute, algorithm):
    # Load file, Original_data.xlsx
    org_data, org_label = LoadData.get_method1_test()

    # Reduce dimension and generate train/test data
    reduced_data = reduce_dimension(org_data, org_label, algorithm)
    # normalized_data = preprocessing.normalize(reduced_data)
    # reduced_data = normalization(reduced_data)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # normalized_data = min_max_scaler.fit_transform(reduced_data)

    # normalized_data = preprocessing.scale(reduced_data)

    normalized_data = Normalize.normalization(reduced_data)

    X_train, X_test, y_train, y_test = train_test_split(
        normalized_data, org_label, test_size=0.3)

    print('<---Test the Label NN Start--->')

    test_output_list = np.array([])
    # 直接投票法，不用LNN
    for test_data, test_label in zip(X_test, y_test):
        lnn_input = get_fnn_output(test_data, fnn_attribute)
        test_output_list = np.append(test_output_list, lnn_input)

    # lnn_input_list = np.array([])
    # for test_data, test_label in zip(X_test, y_test):
    #     lnn_input = get_fnn_output(test_data, fnn_attribute)
    #     lnn_input_list = np.append(lnn_input_list, lnn_input)
    # lnn_input_list = lnn_input_list.reshape(-1, 6)
        # print('label_nn_test(Test)', lnn_input)
    # 產生輸出後再正規化一次
    # lnn_test_input_list = preprocessing.scale(lnn_input_list)

    # lnn_test_input_list = normalization(lnn_input_list)

    # test_output_list = np.array([])
    # for train_data, train_label in zip(X_train, y_train):
    #     lnn_input = get_fnn_output(train_data, fnn_attribute)
    #     # print('lnn_input(Test ALL)', lnn_input)
    #
    #     weight1 = lnn_attribute['Weight1']
    #     weight2 = lnn_attribute['Weight2']
    #     bias = lnn_attribute['Bias']
    #
    #     lnn = LabelNN(lnn_input_size, lnn_hidden_size, lnn_output_size, weight1, weight2, bias, lnn_lr)
    #     test_output = lnn.forward(lnn_input)
    #     test_output_list = np.append(test_output_list, test_output)
    #
    #
    #     # # 直接投票法，不用LNN
    #     # lnn_input = get_fnn_output(train_data, fnn_attribute)
    #     # test_output_list = np.append(test_output_list, lnn_input)

    final_output_list = np.array([])

    # weight1 = lnn_attribute['Weight1']
    # weight2 = lnn_attribute['Weight2']
    # bias = lnn_attribute['Bias']

    # lnn = LabelNN(lnn_input_size, lnn_hidden_size, lnn_output_size, weight1, weight2, bias, lnn_lr)

    # lnn_test_input_list = lnn_input_list.reshape(-1, 6)
    # final_output_list = lnn.forward(lnn_test_input_list)
    # final_output_list = final_output_list.reshape(-1, 6)

    test_output_list = test_output_list.reshape(-1, 6)
    label_pred = LabelNN.label_encode(test_output_list)
    for x, y in zip(test_output_list, y_test):
        print(x, ' ', y)

    # normalized_output = min_max_scaler.fit_transform(test_output_list)
    # print('normalized_output', normalized_output)
    # label_pred = LabelNN.label_encode(normalized_output)

    C_matrix = confusion_matrix(y_test, label_pred)
    C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)

    print('This is the confusion matrix(test_all_model)\n', C_matrix)
    # print(C_matrix)
    # print(C_accuracy)

    print('<---Test the Label NN Successfully--->')
    print('<----------------------------------------------->')
    return C_accuracy


"""
Show the Model, Visulation
Descibe the model

"""


def show_model(mean, stddev, weight):
    data = np.array([])
    output = np.array([])
    for i in range(-10, 10, 1):
        for j in range(-10, 10, 1):
            for k in range(-10, 10, 1):
                tmp = np.append(data, np.array([i, j, k]))
    data = data.reshape(1, -1, 3) / 10
    fnn = FNN(
        fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size,
        mean, stddev, weight, fnn_lr, 1)

    for element in data:
        output = np.append(output, fnn.forward(element))
    ModelScatter.output_scatter_3d(data, output, fnn_threshold1)


if __name__ == '__main__':

    # 目前的降維度法是 tSNE
    for algorithm in dimension_reduce_algorithm:
        start = time.time()
        print('<---Train', algorithm, '--->')

        # Store those values to describe the best model in fnn local training
        fnn_model, fnn_accuracy, fnn_matrix = ([] for _ in range(3))
        pd_header = ['T', 'F']
        for nn in range(1, fnn_label_size + 1, 1):
            fnn, accuracy, matrix = train_local_fnn(nn, algorithm)
            fnn_model.append(fnn)
            # 儲存 fnn model as .json
            rel_path = '../Experiment/method3/Model/FNN/'
            abs_path = os.path.join(os.path.dirname(__file__), rel_path)
            Export.save_fnn_weight(nn, fnn, abs_path)
            fnn_accuracy.append(accuracy)
            fnn_matrix.append(pd.DataFrame(matrix, columns=pd_header, index=pd_header))
        
        for i in range(len(fnn_matrix)):
            rel_path = '../Experiment/method3/Graph/cnf'+str(i)+'.png'
            abs_path = os.path.join(os.path.dirname(__file__), rel_path)
            ConfusionMatrix.plot_confusion_matrix(
                fnn_matrix[i], abs_path, classes=[0,1], title='C'+str(i)+' Cnf')
        


        end = time.time()

        print('All cost time is (' + str(end-start) + ')')
