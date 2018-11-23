import time
import copy
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from Method.LoadData import LoadData
from Method.Normalize import Normalize
from Algorithm.FNN2 import FNN2
from MkGraph.AccuracyPlot import AccuracyPlot
from MkGraph.ErrorPlot import ErrorPlot

"""
# Fuzzy Neural Networks Structure
# Fuzzy Data Set are 24000
"""
fnn_label_size = 6
fnn_input_size = 5
fnn_membership_size = fnn_input_size * fnn_label_size
fnn_rule_size = 6
fnn_output_size = 6
fnn_weight_size = fnn_rule_size * fnn_output_size
fnn_lr = 0.001
fnn_epoch = 1
fnn_random_size = 1

dimension_reduce_algorithm = "LDA"
dim = 5

if __name__ == '__main__':

    # Basic Setting
    start = time.time()
    print('<---Train by', dimension_reduce_algorithm, '--->')
    accuracy = 0.0
    all_accuracy = np.array([])
    record_fnn = FNN2()

    # Load file
    org_data, org_label = LoadData.get_lnn_training_data()

    # Use LDA to reduce dimension
    lda = LinearDiscriminantAnalysis(n_components=5)
    lda.fit(org_data, org_label)
    reduced_data = lda.transform(org_data)

    # Normalized data
    normalized_data = Normalize.normalization(reduced_data)

    # Get Train/Test data set
    X_train, X_test, y_train, y_test = train_test_split(normalized_data, org_label, test_size=0.3)

    # Using the FNN to predict
    print('<---Train the FNN Start--->')
    for i in range(fnn_random_size):
        # Random Generate the mean, standard deviation
        mean = np.array(
            [np.random.uniform(-1, 1) for _ in range(fnn_membership_size)]).reshape(-1, 6)
        stddev = np.array(
            [np.random.uniform(0, 1) for _ in range(fnn_membership_size)]).reshape(-1, 6)
        weight = np.array(
            [np.random.uniform(-1, 1) for _ in range(fnn_weight_size)]).reshape(-1, fnn_rule_size)

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
        # Init Class and Trina the model
        fnn = FNN2(
            fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, fnn_label_size
            , mean, stddev, weight, fnn_lr, 1)
        fnn.training_model(fnn_epoch, X_train, y_train)

        # Test the FNN model, save the one that has the best accuracy
        test_output = fnn.testing_model(X_test)
        test_output = test_output.reshape(-1, 6)
        label_prediction = fnn.label_encode(test_output)
        C_matrix = confusion_matrix(y_test, label_prediction)
        C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)
        all_accuracy = np.append(all_accuracy, C_accuracy)
        print('('+str(i)+'/'+str(fnn_random_size)+')->\n', C_matrix, C_accuracy)
        if C_accuracy > accuracy:
            record_fnn = copy.deepcopy(fnn)
    print('<---Train the FNN Successfully--->')

    # Output relative graph
    abs_path = './Data/Graph/LDA/Best_FNN_error_trend.png'
    # print('ErrorPlot', abs_path)
    ErrorPlot.error_trend(
        'Best_FNN_error_trend', len(record_fnn.error_list), record_fnn.error_list, abs_path)

    abs_path = './Data/Graph/LDA/Accuracy vs FNN.png'
    # print('AccuracyPlot', abs_path)
    AccuracyPlot.build_accuracy_plot(
        'Accuracy vs FNN', np.array([i for i in range(1, len(all_accuracy) + 1, 1)]),
        all_accuracy, abs_path)

    # Output the best FNN
    print('Mean', record_fnn.mean)
    print('Stddev', record_fnn.stddev)
    print('Weight', record_fnn.weight)

    end = time.time()
    print('All cost time is (' + str(end-start) + ')')
