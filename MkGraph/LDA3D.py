import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from Method.LoadData import LoadData

"""
This script output the 2d/3d graph
reduced dimension algorithm is LDA
Loaded file: Data\Labeling\C\LNN_Train_data.xlsx
"""

# Variable
dim = [i+1 for i in range(1, 3, 1)]
color_list = ['r', 'b', 'g', 'y', 'c', 'k']
label_list = ['C'+str(i+1) for i in range(6)]

# Run the experiment from one dimension to five dimension
for sub_dim in dim:
    # Read file LNN_Train_data.xlsx'
    org_data, org_label = LoadData.get_lnn_training_data()

    # Normalize the data
    normalized_data = preprocessing.normalize(org_data)
    # print(normalized_data)

    # Use LDA algorithm to reduce the dimensions
    lda = LinearDiscriminantAnalysis(n_components=sub_dim)
    lda.fit(normalized_data, org_label)
    reduced_data = lda.transform(normalized_data)

    normalized_data = preprocessing.normalize(reduced_data)
    print(normalized_data)

    # Split the original data
    data1, data2, data3, data4, data5, data6 = (np.array([]) for _ in range(6))
    for element, stamp in zip(normalized_data, org_label):
        if stamp == 1:
            data1 = np.append(data1, element)
        elif stamp == 2:
            data2 = np.append(data2, element)
        elif stamp == 3:
            data3 = np.append(data3, element)
        elif stamp == 4:
            data4 = np.append(data4, element)
        elif stamp == 5:
            data5 = np.append(data5, element)
        elif stamp == 6:
            data6 = np.append(data6, element)
        else:
            print('Error plot')

    if sub_dim == 2:
        plt.figure(figsize = (8, 6), dpi = 100)
        plt.subplot(111)

        data1 = data1.reshape(-1, 2).T
        data2 = data2.reshape(-1, 2).T
        data3 = data3.reshape(-1, 2).T
        data4 = data4.reshape(-1, 2).T
        data5 = data5.reshape(-1, 2).T
        data6 = data6.reshape(-1, 2).T

        plt.scatter(data1[0], data1[1], color=color_list[0], label=label_list[0])
        plt.scatter(data2[0], data2[1], color=color_list[1], label=label_list[1])
        plt.scatter(data3[0], data3[1], color=color_list[2], label=label_list[2])
        plt.scatter(data4[0], data4[1], color=color_list[3], label=label_list[3])
        plt.scatter(data5[0], data5[1], color=color_list[4], label=label_list[4])
        plt.scatter(data6[0], data6[1], color=color_list[5], label=label_list[5])

        # Set attribute
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('NN Scatter2D')
        plt.legend('upper right')

    else:
        fig = plt.figure()
        ax = Axes3D(fig)

        data1 = data1.reshape(-1, 3).T
        data2 = data2.reshape(-1, 3).T
        data3 = data3.reshape(-1, 3).T
        data4 = data4.reshape(-1, 3).T
        data5 = data5.reshape(-1, 3).T
        data6 = data6.reshape(-1, 3).T

        # Scatter graph
        ax.scatter(data1[0], data1[1], data1[2], color=color_list[0])
        ax.scatter(data2[0], data2[1], data2[2], color=color_list[1])
        ax.scatter(data3[0], data3[1], data3[2], color=color_list[2])
        ax.scatter(data4[0], data4[1], data4[2], color=color_list[3])
        ax.scatter(data5[0], data5[1], data5[2], color=color_list[4])
        ax.scatter(data6[0], data6[1], data6[2], color=color_list[5])

        # Set attribute
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('NN Scatter3D')

    # Output the graph
    rel_path = '../Data/Graph/LDA_Graph_' + str(sub_dim) + '.png'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    plt.savefig(abs_path)
    plt.show()
