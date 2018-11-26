"""
This script train the NN1, NN5, NN6 with Fuzzy Neural Networks
And train the NN2, NN3, NN4 with Deep Neural Networks
"""

import os
import time
import copy
import datetime
import numpy as np

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn import preprocessing

from Method.LoadData import LoadData
from Algorithm.FNN import FNN
from Algorithm.Keras.DNN import DNN
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
fnn_epoch = 1
fnn_random_size = 1


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
dimension_reduce_algorithm = ['Isomap']


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
    org_label = np.array([1 if element == nn else -1 for element in org_label])

    # Reduce dimension and generate train/test data
    reduced_data = reduce_dimension(org_data, algorithm)
    # normalized_data = preprocessing.normalize(reduced_data)

    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_data = min_max_scaler.fit_transform(org_data)

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
        org_path = '.\\Data\\Graph\\'
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
Train NN with DNN
"""


def train_local_dnn(nn):
    # Train the DNN
    print('<---Train the DNN' + str(nn) + ' Start--->')
    org_data, org_label = LoadData.get_fnn_training_data(nn)
    org_label = np.array([[1, 0] if element == nn else [0, 1] for element in org_label])
    print(org_label)

    # normalized_data = preprocessing.normalize(org_data)

    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(org_data)

    # label need to convert by OneHotEncoding
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, org_label, test_size=0.3)
    # y_train_onehot = np_utils.to_categorical(y_train)
    # y_test_onehot = np_utils.to_categorical(y_test)

    # print("x_Train", X_train.shape)
    # print("x_Train", X_train)

    # print("y_Train", y_train_onehot.shape)
    # print("y_Train", y_train_onehot)

    # print("y_Train", y_train.shape)
    # print("y_Train", y_train)

    # print("x_Test", X_test.shape)
    # print("x_Test", X_test)

    # print("y_Test", y_test_onehot.shape)
    # print("y_Test", y_test_onehot)

    # print("y_Test", y_test.shape)
    # print("y_Test", y_test)

    DNN.dnn_train(nn, X_train, y_train, X_test, y_test)
    print('<---Train the DNN' + str(nn) + ' Successfully--->')
    print('<----------------------------------------------->')


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
            # If nn == 2, 3, 4
            # We use DNN to train
            if 2 <= nn <= 4:
                train_local_dnn(nn)
                continue

            else:
                continue
                p1, p2, p3, p4, p5 = train_local_fnn(nn, algorithm)
                fnn_mean.append(p1)
                fnn_stddev.append(p2)
                fnn_weight.append(p3)
                fnn_accuracy.append(p4)
                fnn_matrix.append(pd.DataFrame(p5, columns=pd_header, index=pd_header))

        print('fnn_mean', fnn_mean)
        print('fnn_stddev', fnn_stddev)
        print('fnn_weight', fnn_weight)