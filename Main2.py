"""
Method 3, FNN + DNN(Keras)
"""


import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.layers import SimpleRNN, LSTM ,Reshape
from keras.utils import np_utils
from numpy import genfromtxt
from keras.models import load_model

from Method.LoadData import LoadData
from Method.Export import Export
from Method.Normalize import Normalize
from Method.ReducedAlgorithm import ReducedAlgorithm as ra
from Algorithm.FNN import FNN
from MkGraph.AccuracyPlot import AccuracyPlot
from MkGraph.ErrorPlot import ErrorPlot
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
fnn_epoch = 4
fnn_random_size = 100

fnn_threshold = 0.0

fnn_threshold1 = 0.0
fnn_threshold2 = 0.0
fnn_threshold3 = 0.0
fnn_threshold4 = 0.0
fnn_threshold5 = 0.0
fnn_threshold6 = 0.0

"""Dimension reduce algorithm"""
dimension_reduce_algorithm = ['tSNE']
def train_local_fnn(algorithm, X_train, X_test, y_train, y_test):
    accuracy = 0.0
    matrix = np.array([])
    fnn_copy = FNN()
    # This variable is used to store the all accuracy
    all_nn_accuracy = np.array([])
    for i in range(fnn_random_size):
        # Random Generate the mean, standard deviation
        mean = np.array(
            [np.random.uniform(-1, 1) for _ in range(fnn_membership_size)])
        stddev = np.array(
            [np.random.uniform(0, 1) for _ in range(fnn_membership_size)])
        weight = np.array(
            [np.random.uniform(-1, 1) for _ in range(fnn_rule_size)])

        fnn = FNN(
            fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size,
            mean, stddev, weight, fnn_lr, 1)
        fnn.training_model(fnn_epoch, X_train, y_train)

        # Test the FNN model, save the one that has the best accuracy
        test_output = fnn.testing_model(X_test)
        label_pred = np.array([1 if value > fnn_threshold else 0 for value in test_output])

        C_matrix = confusion_matrix(y_test, label_pred)
        C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)
        all_nn_accuracy = np.append(all_nn_accuracy, C_accuracy)
        if C_accuracy > accuracy:
            accuracy = copy.deepcopy(C_accuracy)
            fnn_copy = copy.deepcopy(fnn)
            matrix = copy.deepcopy(C_matrix)

    # Choose the best FNN to Plot error trend
    # rel_path = './Experiment/Method3/Graph/Best_FNN_'+str(nn)+'_error_trend.png'
    # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    # ErrorPlot.error_trend('Best_FNN_'+str(nn)+'_error_trend',
    #                       len(fnn_copy.error_list), fnn_copy.error_list, abs_path)

    # Choose the best Accuracy to Plot
    # rel_path = './Experiment/Method3/Graph/Accuracy vs FNN'+str(nn)+'.png'
    # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    # AccuracyPlot.build_accuracy_plot(
    #    'Accuracy vs FNN' +
    #    str(nn), np.array([i for i in range(1, len(all_nn_accuracy) + 1, 1)]),
    #    all_nn_accuracy, abs_path)

    return fnn_copy, accuracy, matrix

def show_train_history(train_history, train, validation, file_name):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(file_name)
    plt.show()


def label_convert(data):
    result = np.array([])
    for element in data:
        if element == 'C1_0':
            result = np.append(result, 0)
        elif element == 'C1_1':
            result = np.append(result, 1)
        elif element == 'C2_0':
            result = np.append(result, 2)
        elif element == 'C2_1':
            result = np.append(result, 3)
        elif element == 'C3_0':
            result = np.append(result, 4)
        elif element == 'C3_1':
            result = np.append(result, 5)
        elif element == 'C4_0':
            result = np.append(result, 6)
        elif element == 'C4_1':
            result = np.append(result, 7)
        elif element == 'C5_0':
            result = np.append(result, 8)
        elif element == 'C5_1':
            result = np.append(result, 9)
        elif element == 'C6_0':
            result = np.append(result, 10)
        elif element == 'C6_1':
            result = np.append(result, 11)
        else:
            print('Error 139')
    return result


def prediction_convert(data):
    result = np.array([])
    for num in data:
        if 0 <= num < 2:
            result = np.append(result, 0)
        elif 2 <= num < 4:
            result = np.append(result, 1)
        elif 4 <= num < 6:
            result = np.append(result, 2)
        elif 6 <= num < 8:
            result = np.append(result, 3)
        elif 8 <= num < 10:
            result = np.append(result, 4)
        elif 10 <= num < 12:
            result = np.append(result, 5)
        else:
            print('Error 159')
    return result

def onehot_convert(data):
    result = np.array([])
    for element in data:
        num = np.argmax(element)
        if 0 <= num < 2:
            result = np.append(result, 0)
        elif 2 <= num < 4:
            result = np.append(result, 1)
        elif 4 <= num < 6:
            result = np.append(result, 2)
        elif 6 <= num < 8:
            result = np.append(result, 3)
        elif 8 <= num < 10:
            result = np.append(result, 4)
        elif 10 <= num < 12:
            result = np.append(result, 5)
        else:
            print('Error 179')
    return result

def train_keras_lnn(nn_array, org_data, org_label, algorithm):
    """Get the fnn output and input the lnn"""
    fnn_output = np.array([])
    for name in nn_array:
        print('<---nn -> ', name, '--->')
        rel_path = './Experiment/Method3/FNNModel/'+name+'.json'
        abs_path = os.path.join(os.path.dirname(__file__), rel_path)
        attribute = LoadData.load_fnn_weight(abs_path)
        mean = np.asarray(attribute['Mean'])
        stddev = np.asarray(attribute['Stddev'])
        weight = np.asarray(attribute['Weight'])
        # Test the FNN
        fnn = FNN(
            fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, mean, stddev, weight, fnn_lr, 1)
        result = fnn.testing_model(org_data)
        fnn_output = np.append(fnn_output, result)

    fnn_output = fnn_output.reshape(len(nn_array), -1).T

    # fnn_label = np.array([int(e[1:2])-1 for e in org_label])
    print('org_label', org_label)
    fnn_label = label_convert(org_label)
    X_train, X_test, y_train, y_test = train_test_split(fnn_output, fnn_label, test_size=0.3, random_state=42)
    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)

    # Construct the lnn
    y_trainOneHot = np_utils.to_categorical(y_train)
    y_testOneHot = np_utils.to_categorical(y_test)

    model = Sequential()
    model.add(Dense(units=32, input_dim=12))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(units=12, kernel_initializer='normal', activation='softmax'))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
    model.summary()

    train_history = model.fit(x=X_train, y=y_trainOneHot, validation_split=0.2, epochs=30, batch_size=200, verbose=2)
    show_train_history(train_history, 'mean_squared_error', 'val_mean_squared_error', 'mean_squared_error.png')
    show_train_history(train_history, 'loss', 'val_loss', 'loss.png')

    scores = model.evaluate(X_test, y_testOneHot)
    print('scores', scores)

    prediction = model.predict(X_test)
    for x, y in zip(prediction[:10], y_testOneHot[:10]):
        print(x, ' ', y)

    prediction = model.predict_classes(X_test)
    y_pred = prediction_convert(prediction)
    yy = onehot_convert(y_testOneHot)

    print(set(y_pred))
    print(set(yy))

    cnf_matrix = confusion_matrix(yy, y_pred)
    print('accuracy_score', accuracy_score(yy, y_pred))
    print('cnf_matrix\n', cnf_matrix)
    rel_path = './Experiment/method3/Graph/cnf_lnn.png'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    plt.figure(figsize=(8, 6), dpi=200)
    ConfusionMatrix.plot_confusion_matrix(cnf_matrix, abs_path,
     classes=list(set(y_pred)), title='Final Model Confusion matrix')
    # 模型存起來
    # path_name = 'FinalModel.h5'
    # model.save(path_name)


if __name__ == '__main__':
    for algorithm in dimension_reduce_algorithm:
        start = time.time()

        # fnn_accuracy, fnn_matrix = ([] for _ in range(2))
        # # nn_category = {'C1': 0, 'C2': 2, 'C3': 2, 'C4': 2, 'C5': 0, 'C6': 0}
        # nn_category = {'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2, 'C5': 2, 'C6': 2}
        # print('<---Part1, Train FNN(C1-C6)--->')

        # for key, value in nn_category.items():
        #     number = 1 if value == 0 else value
        #     for num in range(number):
        #         print(key, ' ', num, ' ', number)
        #         if value == 0:
        #             name = str(key)
        #         else:
        #             name = str(key)+'_'+str(num)
        #         print('<---name is ', name, '--->')
        #         print('<---Train the FNN' + name + ' Start--->')
        #         org_data, org_label = LoadData.get_method3_fnn_train(name)
        #         org_label = np.array([1 if element == name else 0 for element in org_label])
        #         X_train, X_test, y_train, y_test = train_test_split(org_data, org_label, test_size=0.3)
        #         fnn, accuracy, matrix = train_local_fnn(algorithm, X_train, X_test, y_train, y_test)

        #         rel_path = './Experiment/Method3/FNNModel/'
        #         abs_path = os.path.join(os.path.dirname(__file__), rel_path)
        #         Export.save_fnn_weight(name, fnn, abs_path)

        #         fnn_accuracy.append(accuracy)
        #         fnn_matrix.append(matrix)

        #         print('<---Train the FNN' + name + ' Successfully--->')
        #         print('<----------------------------------------------->')

        # print('fnn_matrix', fnn_matrix)
        # print('fnn_accuracy', fnn_accuracy)
            #for i in range(len(fnn_matrix)):
                #rel_path = './Experiment/method3/Graph/cnf'+str(i)+'.png'
               # abs_path = os.path.join(os.path.dirname(__file__), rel_path)
                #ConfusionMatrix.plot_confusion_matrix(
                   # fnn_matrix[i], abs_path, classes=[0,1], title='C'+str(i)+' Cnf')

        print('<---Part2, Keras Networks(LNN)--->')
        # org_data, org_label = LoadData.get_lnn_training_data()
        org_data, org_label = LoadData.get_method3_test()
        # reduced_data = ra.pca(org_data, fnn_input_size)
        # normalized_data = Normalize.normalization(reduced_data)
        # nn_category = [i for i in range(1, 7, 1)]

        nn_array = ['C1_0', 'C1_1', 'C2_0', 'C2_1',
                    'C3_0', 'C3_1', 'C4_0', 'C4_1',
                    'C5_0', 'C5_1', 'C6_0', 'C6_1']
        train_keras_lnn(nn_array, org_data, org_label, algorithm)

        end = time.time()
        print('All cost time is (' + str(end-start) + ')')
