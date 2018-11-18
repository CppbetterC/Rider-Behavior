"""
由 Original_data 統計出來個別標籤的數量
c1, 21
c2, 5578
c3, 248
c4, 418
c5, 35
c6, 5
"""

import os, sys
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
from Algorithm.FNN import FNN
from Algorithm.LabelNN import LabelNN
from MkGraph.AccuracyPlot import AccuracyPlot
from MkGraph.ErrorPlot import ErrorPlot

"""
# Fuzzy Neural Networks Structure
# Fuzzy Data Set are 1000
"""
fnn_label_size = 6
fnn_input_size = 3
fnn_membership_size = fnn_input_size * fnn_label_size
fnn_rule_size = 6
fnn_output_size = 1
fnn_lr = 0.0001
fnn_threshold = 0.0
fnn_epoch = 10
fnn_random_size = 100


"""
# Label Neural Network Structure
# Label NN Data Set are 6000
"""
lnn_input_size = 6
lnn_hidden_size = 6
lnn_output_size = 6
lnn_lr = 0.0001
lnn_epoch = 1
lnn_random_size = 150

"""
Dimension reduce algorithm
"""
# dimension_reduce_algorithm = ['LLE', 'PCA', 'Isomap']
dimension_reduce_algorithm = ['PCA']

# # Normalization the data
# # The interval is between -1 and 1
# def normalization(data):
#     new_data = np.array([])
#     tmp = data.T
#     length = len(tmp[0])
#     for array in tmp:
#         sub_array = []
#         max_value = max(array)
#         min_value = min(array)
#         for element in array:
#             sub_array.append(2 * ((element - min_value) / (max_value - min_value)) - 1)
#         new_data = np.append(new_data, sub_array)
#     new_data = new_data.reshape(-1, length).T
#     return new_data


# Create a storage to new picture
def makedir(path, algorithm_name):
    # Get the timestamp
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    try:
        os.mkdir(path + algorithm_name)
        os.chdir(path + algorithm_name)
    except FileExistsError:
        os.chdir(path + algorithm_name)
        print("<---FileExistsError(Main.py)--->")

    # with open('README.txt', 'w', encoding='utf-8') as f:
    #     f.writelines('timestamp ->' + timestamp)
    #     f.writelines('algorithm_name' + algorithm_name)

    # return os.getcwd()


# Using the LLE(Locally Linear Embedding)
# To reduce the dimension
def reduce_dimension(data, algorithm_name):
    dim = fnn_input_size
    data_new = np.array([])

    if algorithm_name == 'LLE' or 'lle':
        # LLE
        embedding = LocallyLinearEmbedding(n_components=dim, eigen_solver='dense')
        data_new = embedding.fit_transform(data)

    elif algorithm_name == 'PCA' or 'pca':
        # PCA
        pca = PCA(n_components=dim)
        pca.fit(data)
        data_new = pca.transform(data)

    elif algorithm_name == 'Isomap' or 'isomap':
        # Isomap
        embedding = Isomap(n_components=dim)
        data_new = embedding.fit_transform(data)
    else:
        print('None dimension reduced')

    return data_new


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
    record_fnn = FNN()
    loss_list = np.array([])

    # This variable is used to store the all accuracy
    all_nn_accuracy = np.array([])

    # Load file FNN_Train_data_' + str(num) + '.xlsx
    org_data, org_label = LoadData.get_fnn_training_data(nn)
    org_label = np.array([1 if element == nn else 0 for element in org_label])

    # Reduce dimension and generate train/test data
    reduced_data = reduce_dimension(org_data, algorithm)
    normalized_data = preprocessing.normalize(reduced_data)
    # reduced_data = normalization(reduced_data)

    X_train, X_test, y_train, y_test = train_test_split(normalized_data, org_label, test_size=0.3)
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
        label_pred = np.array([1 if element >= fnn_threshold else 0 for element in test_output])

        # print(y_test.shape)
        # print(label_pred.shape)
        # print(y_test)
        # print(label_pred)

        C_matrix = confusion_matrix(y_test, label_pred)
        C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)
        all_nn_accuracy = np.append(all_nn_accuracy, C_accuracy)

        # print(C_matrix)
        # print(C_accuracy)
        if C_accuracy > accuracy:
            accuracy = copy.deepcopy(C_accuracy)
            nn_mean = copy.deepcopy(fnn.mean)
            nn_stddev = copy.deepcopy(fnn.stddev)
            nn_weight = copy.deepcopy(fnn.weight)
            matrix = copy.deepcopy(C_matrix)
            record_fnn = copy.deepcopy(fnn)
            loss_list = copy.deepcopy(fnn.loss_list)

        """
        Every error trend graph will output
        Output the Error Plot to observe trend
        """
        # rel_path = './Data/Graph/' + str(i) + '_FNN_' + str(nn) + '_error_trend.png'
        # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
        # ErrorPlot.error_trend(
        #     str(i) + '_FNN_' + str(nn) + '_error_trend', len(fnn.error_list), fnn.error_list, abs_path)

    print('<---Train the FNN' + str(nn) + ' Successfully--->')
    print('<----------------------------------------------->')

    # print('1_目錄:', os.getcwd())

    # First Time, you need to create a folder
    if nn == 1:
        org_path = './Data/Graph/'
        makedir(org_path, algorithm)
    # else:
    #     os.chdir('./Data/Graph/' + dimension_reduce_algorithm)
    # print('2_目錄:', os.getcwd())

    # Choose the best FNN to Plot error trend
    # rel_path = org_path + 'Best_FNN_' + str(nn) + '_error_trend.png'
    # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    abs_path = os.getcwd() + '\\Best_FNN_' + str(nn) + '_error_trend.png'
    # print('ErrorPlot', abs_path)
    ErrorPlot.error_trend(
        'Best_FNN_' + str(nn) + '_error_trend', len(record_fnn.error_list), record_fnn.error_list, abs_path)

    abs_path = os.getcwd() + '\\Best_FNN_' + str(nn) + '_loss_trend.png'
    # Choose the best FNN to Plot loss on every epoch
    ErrorPlot.loss_trend(
        'Best_FNN_' + str(nn) + '_loss_trend', len(loss_list), loss_list, abs_path)

    # Choose the best Accuracy to Plot
    # rel_path = org_path + 'Accuracy vs FNN' + str(nn) + '.png'
    # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    abs_path = os.getcwd() + '\\Accuracy vs FNN' + str(nn) + '.png'
    # print('AccuracyPlot', abs_path)
    AccuracyPlot.build_accuracy_plot(
        'Accuracy vs FNN'+str(nn), np.array([i for i in range(1, len(all_nn_accuracy) + 1, 1)]),
        all_nn_accuracy, abs_path)

    return nn_mean, nn_stddev, nn_weight, accuracy, matrix


"""
Be used to output those values of the fnn
input those values to fnn6
"""


def get_fnn_output(data, fnn_attribute):
    forward_output_list = np.array([])
    for i in range(0, 6, 1):
        # print('<--- Print the FNN ' + str(nn) + ' Output--->')
        mean = fnn_attribute['Mean'][i]
        stddev = fnn_attribute['Stddev'][i]
        weight = fnn_attribute['Weight'][i]
        fnn = FNN(
            fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, mean, stddev, weight, fnn_lr, 1)
        forward_output = fnn.forward(data)
        forward_output_list = np.append(forward_output_list, forward_output)
    return forward_output_list


"""
Train the NN to distinguish behavior label
Call class LabelNN to train

para1 -> Those recorded attribute from the method/(train_local_fnn)
para2 -> which algorithm is used to reduce algorithm for lnn
"""


def train_label_nn(fnn_attribute, algorithm):
    # Declare variables
    nn_weight1, nn_weight2, nn_bias = (0.0 for _ in range(3))
    accuracy = 0.0
    matrix = np.array([])
    record_lnn = LabelNN()

    # This variable is used to store the all accuracy
    all_nn_accuracy = np.array([])

    # Load file LNN_Train_data.xlsx
    org_data, org_label = LoadData.get_lnn_training_data()

    # Reduce dimension and generate train/test data
    reduced_data = reduce_dimension(org_data, algorithm)

    normalized_data = preprocessing.normalize(reduced_data)
    # reduced_data = normalization(reduced_data)
    X_train, X_test, y_train, y_test = train_test_split(normalized_data, org_label, test_size=0.3)

    # print('X_train', X_train)
    # print('X_test', X_test)

    # Label Neural Networks Structure
    weight1_size = 36
    weight2_size = 36
    bias_size = 6

    # Save the one that has the best accuracy
    lnn_input_list = np.array([])
    for test_data, test_label in zip(X_test, y_test):
        lnn_input = get_fnn_output(test_data, fnn_attribute)
        lnn_input_list = np.append(lnn_input_list, lnn_input)
        lnn_input_list = lnn_input_list.reshape(-1, 6)
        # print('label_nn_test(Test)', lnn_input)

    # Train the Label NN start
    print('<---Train the Label NN Start--->')
    for _ in range(lnn_random_size):

        weight1 = \
            np.array([np.random.uniform(-1, 1) for _ in range(weight1_size)]).reshape(-1, 6)
        weight2 = \
            np.array([np.random.uniform(-1, 1) for _ in range(weight2_size)]).reshape(-1, 6)
        bias = \
            np.array([np.random.uniform(-1, 1) for _ in range(bias_size)])

        lnn = LabelNN(lnn_input_size, lnn_hidden_size, lnn_output_size, weight1, weight2, bias, lnn_lr)

        for train_data, train_label in zip(X_train, y_train):
            # Calculate the input of the LNN
            # By getting the output the FNN1 ~ FNN6
            lnn_input = get_fnn_output(train_data, fnn_attribute)

            # print('lnn_input', lnn_input)

            # print('lnn_input(Train)', lnn_input)
            try:
                lnn.training_model(lnn_epoch, lnn_input, train_label)

            except OverflowError:
                print("<---Main.py(Something error had happen in train lnn)--->")
                break
            except ZeroDivisionError:
                print("<---Main.py(Something error had happen in train lnn)--->")
                break

        # Test the FNN model,
        # Encoding the label NN
        # Make the confusion matrix
        try:
            test_output = lnn.testing_model(lnn_input_list)

        except OverflowError:
            print("<---Main.py(Something error had happen in test lnn)--->")
            continue

        label_pred = LabelNN.label_encode(test_output)
        C_matrix = confusion_matrix(y_test, label_pred)
        C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)

        # Record the single accuracy
        all_nn_accuracy = np.append(all_nn_accuracy, C_accuracy)
        # print(C_matrix)
        # print(C_accuracy)

        if C_accuracy > accuracy:
            accuracy = copy.deepcopy(C_accuracy)
            nn_weight1 = copy.deepcopy(lnn.weight1)
            nn_weight2 = copy.deepcopy(lnn.weight2)
            nn_bias = copy.deepcopy(lnn.bias)
            matrix = copy.deepcopy(C_matrix)
            record_lnn = copy.deepcopy(lnn)

        """
        Every error trend graph will output
        Output the Error Plot to observe trend
        """
        # rel_path = './Data/Graph/' + str(i) + '_LNN_error_trend.png'
        # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
        # ErrorPlot.error_trend(
        #     str(i) + '_LNN_error_trend', len(lnn.error_list), lnn.error_list)

    print('<---Train the Label NN Successfully--->')
    print('<----------------------------------------------->')

    # # Create a folder
    # org_path = './Data/Graph/'
    # org_path = makedir(org_path, dimension_reduce_algorithm)

    # print('3_目錄', os.getcwd())

    abs_path = os.getcwd()
    # Choose the best LNN to Plot error trend
    ErrorPlot.mul_error_trend(
        'Best_LNN_error_trend', len(record_lnn.error_list), record_lnn.error_list, abs_path)

    # Choose the best Accuracy to Plot
    # rel_path = org_path + 'Accuracy vs LNN.png'
    # abs_path = os.path.join(os.path.dirname(__file__), rel_path)

    abs_path = os.getcwd() + '\\Accuracy vs LNN.png'
    AccuracyPlot.build_accuracy_plot(
        'Accuracy vs LNN', np.array([i for i in range(1, len(all_nn_accuracy) + 1, 1)]),
        all_nn_accuracy, abs_path)

    return nn_weight1, nn_weight2, nn_bias, accuracy, matrix


"""
Test all model
"""


def test_all_model(fnn_attribute, lnn_attribute, algorithm):
    # Load file, Original_data.xlsx
    org_data, org_label = LoadData.get_test_data()

    # Reduce dimension and generate train/test data
    reduced_data = reduce_dimension(org_data, algorithm)
    normalized_data = preprocessing.normalize(reduced_data)
    # reduced_data = normalization(reduced_data)

    X_train, X_test, y_train, y_test = train_test_split(normalized_data, org_label, test_size=0.3)

    print('<---Test the Label NN Start--->')
    test_output_list = np.array([])
    for train_data, train_label in zip(X_train, y_train):
        lnn_input = get_fnn_output(train_data, fnn_attribute)
        # print('lnn_input(Test ALL)', lnn_input)

        weight1 = lnn_attribute['Weight1']
        weight2 = lnn_attribute['Weight2']
        bias = lnn_attribute['Bias']

        lnn = LabelNN(lnn_input_size, lnn_hidden_size, lnn_output_size, weight1, weight2, bias, lnn_lr)
        test_output = lnn.forward(lnn_input)
        test_output_list = np.append(test_output_list, test_output)

    test_output_list = test_output_list.reshape(-1, 6)
    label_pred = LabelNN.label_encode(test_output_list)
    C_matrix = confusion_matrix(y_train, label_pred)
    C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)

    print('This is the confusion matrix(test_all_model)\n', C_matrix)
    # print(C_matrix)
    # print(C_accuracy)

    print('<---Test the Label NN Successfully--->')
    print('<----------------------------------------------->')
    return C_accuracy


if __name__ == '__main__':

    for algorithm in dimension_reduce_algorithm:

        start = time.time()

        # Store those values to describe the best model in fnn local training
        fnn_mean, fnn_stddev, fnn_weight, fnn_accuracy, fnn_matrix =\
            ([] for _ in range(5))

        """
        Start to train FNN (Fuzzy Neural Networks)
        We will train six FNN (FNN1 ~ FNN6)
        para1 -> mean, 
        para2 -> standard deviation
        para3 -> weight, 
        para4 -> accuracy
        para5 -> confusion matrix
        """
        pd_header = ['T', 'F']
        for nn in range(1, fnn_label_size + 1, 1):
            p1, p2, p3, p4, p5 = train_local_fnn(nn, algorithm)
            fnn_mean.append(p1)
            fnn_stddev.append(p2)
            fnn_weight.append(p3)
            fnn_accuracy.append(p4)
            fnn_matrix.append(pd.DataFrame(p5, columns=pd_header, index=pd_header))

        # Store and print those values
        header = ['Mean', 'Stddev', 'Weight', 'Local Accuracy']
        # idx = ['Fnn1', 'Fnn2', 'Fnn3', 'Fnn4', 'Fnn5', 'Fnn6']
        fnn_statistics = \
            {header[0]: fnn_mean, header[1]: fnn_stddev, header[2]: fnn_weight, header[3]: fnn_accuracy}
        # print('fnn_statistics\n', fnn_statistics)
        for key, item in fnn_statistics.items():
            print(key, '=>', item)

        fnn_label = ['FNN1', 'FNN2', 'FNN3', 'FNN4', 'FNN5', 'FNN6']
        for df, nn_label in zip(fnn_matrix, fnn_label):
            print('confusion matrix', nn_label)
            print(df)
            print('<----------------------------------------------->')

        """
        After training six FNN
        We will train the seventh neural networks
        This neural network is used to distinguish the behavior label
        We record the mean, stddev, weight, the method of the reduced dimension
        We will use above those values and input the data of the LNN_Train_data.xlsx
        """
        lnn_weight1, lnn_weight2, lnn_bias, lnn_accuracy, lnn_matrix = train_label_nn(fnn_statistics, algorithm)
        header = ['Weight1', 'Weight2', 'Bias', 'Label Accuracy']
        lnn_statistics = \
            {header[0]: lnn_weight1, header[1]: lnn_weight2, header[2]: lnn_bias, header[3]: lnn_accuracy}
        # print('lnn_statistics\n', lnn_statistics)
        for key, item in lnn_statistics.items():
            print(key, '=>', item)

        # Output the Confusion Matirx
        print('<----------------------------------------------->')
        pd_header = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        print('confusion matrix\n', pd.DataFrame(lnn_matrix, columns=pd_header, index=pd_header))
        print('<----------------------------------------------->')

        # Use the LNN_Train.xlsx to test all model
        # All model contain FNN1 ~ FNN6 and LNN
        model_accuracy = test_all_model(fnn_statistics, lnn_statistics, algorithm)
        print('Model Accuracy', model_accuracy)

        end = time.time()

        print('All cost time is (' + str(end-start) + ')')
