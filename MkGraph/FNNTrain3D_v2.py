"""
This script read the method2/FNN_Train_data
reduced dimension algorithm is tSNE
Output the 3D scatter
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Method.LoadData import LoadData

color_list = ['r', 'b', 'g', 'y', 'c', 'k']

# Best -> tSNE
algorithm = 'tSNE'
dimension = 3

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

# Run the experiment from one dimension to five dimension
for nn in nn_category:
    # Read file LNN_Train_data.xlsx'
    org_data, org_label = LoadData.get_method2_fnn_train(nn)
    print('org_data', org_data)
    print('org_label', org_label)

    data1, data2 = (np.array([]) for _ in range(2))
    for element, stamp in zip(org_data, org_label):
        print(element, stamp)
        if stamp == nn:
            data1 = np.append(data1, element)
        else:
            data2 = np.append(data2, element)

    # Make graph
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = Axes3D(fig)

    data1 = data1.reshape(-1, 3).T
    data2 = data2.reshape(-1, 3).T

    # Scatter graph
    ax.scatter(data1[0], data1[1], data1[2], color=color_list[0])
    ax.scatter(data2[0], data2[1], data2[2], color=color_list[1])

    # Set attribute
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('NN Scatter3D')

    # Output the graph
    rel_path = '../Experiment/Graph/' + algorithm + '_Graph_FNN' + str(nn) + '_' + '.png'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    plt.savefig(abs_path)
    plt.show()
    # plt.ion()
    # plt.pause(5)
    # plt.close()
