"""
觀察類神經，調整 Threshold
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import confusion_matrix

from Method.LoadData import LoadData
from MkGraph.ConfusionMatrix import ConfusionMatrix
from Algorithm.FNN import FNN

all_label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
cluster_num = {'C1': 6, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5, 'C6': 4}
nn_category = np.array([])
for element in all_label:
    if cluster_num[element] == 0:
        nn_category = np.append(nn_category, element)
    else:
        for num in range(cluster_num[element]):
            nn_category = np.append(nn_category, element + '_' + str(num))
print('nn_category', nn_category)


fnn_label_size = 6
fnn_input_size = 3
fnn_membership_size = fnn_input_size * fnn_label_size
fnn_rule_size = 6
fnn_output_size = 1
fnn_lr = 0.001
fnn_epoch = 1
fnn_random_size = 1

fnn_threshold = 0.0

for nn in nn_category:
    # Load the json
    rel_path = '../Experiment/Model/FNN/'+str(nn)+'.json'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    attribute = LoadData.load_fnn_weight(abs_path)
    # print(attribute)

    # Load the test data
    org_data, org_label = LoadData.get_method2_fnn_train(nn)
    print('org_data.shape', org_data.shape)
    print('org_label.shape', org_label)

    mean = np.asarray(attribute['Mean'])
    stddev = np.asarray(attribute['Stddev'])
    weight = np.asarray(attribute['Weight'])
    # Test the FNN
    fnn = FNN(
        fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, mean, stddev, weight, fnn_lr, 1)

    output = fnn.testing_model(org_data)

    # 全部訓練資料的 confusion matrix
    # Hot map
    rel_path = '../Experiment/Graph/test/CM/CM'+str(nn)+'.png'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    y_test = [1 if label == nn else 0 for label in org_label]
    y_pred = [1 if value > fnn_threshold else 0 for value in output]
    # print(y_test)
    # print(y_pred)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    ConfusionMatrix.plot_confusion_matrix(cnf_matrix, abs_path,
                                          classes=list(set(y_test)), title=nn+' Confusion matrix')

    error_input, correct_input = (np.array([]) for _ in range(2))
    for x, y in zip(org_data, output):
        if y <= fnn_threshold:
            if len(error_input) == 0:
                error_input = x.reshape(-1, 3)
            else:
                error_input = np.concatenate((error_input, x.reshape(-1, 3)), axis=0)
        else:
            if len(correct_input) == 0:
                correct_input = x.reshape(-1, 3)
            else:
                correct_input = np.concatenate((correct_input, x.reshape(-1, 3)), axis=0)

    # print('len(correct_input)', len(correct_input))
    # print('len(error_input)', len(error_input))

    # Scatter
    rel_path = '../Experiment/Graph/test/Scatter/Scatter'+str(nn)+'.png'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    correct_data = correct_input.T
    error_data = error_input.T
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = Axes3D(fig)
    ax.scatter(correct_data[0], correct_data[1], correct_data[2], color='b')
    ax.scatter(error_data[0], error_data[1], error_data[2], color='r')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(abs_path)
    plt.show()

    # Bar
    rel_path = '../Experiment/Graph/test/Bar/Bar'+str(nn)+'.png'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    x_axis = ['Up Fnn threshold', 'Down Fnn threshold']
    y_axis = [len(correct_input), len(error_input)]
    plt.bar(x_axis, y_axis)
    plt.savefig(abs_path)
    plt.show()
