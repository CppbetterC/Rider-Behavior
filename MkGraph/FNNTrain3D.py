import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing

from Method.LoadData import LoadData
from Method.ReducedAlgorithm import ReducedAlgorithm as ra

"""
This script output the 2d/3d graph
reduced dimension algorithm is LDA
Loaded file: Data\Labeling\C\LNN_Train_data.xlsx
"""

# Variable
dim = 3
color_list = ['r', 'b', 'g', 'y', 'c', 'k']
# reduced_algorithm =\
#     ['lle', 'pca', 'mds', 'ProbPCA', 'FactorAnalysis',
#      'Isomap', 'HessianLLE', 'LTSA', 'KernelPCA', 'tSNE', 'KernelLDA', 'NCA', 'LMNN'
#
#      'DiffusionMaps','SNE', 'SymSNE',
#      'NPE', 'LPP', 'SPE', 'LLTSA','CCA', 'MVU', 'LandmarkMVU', 'FastMVU', 'LLC',
#      'ManifoldChart', 'CFA', 'GPLVM', 'Autoencoder', , 'MCML', ]

reduced_algorithm = ['MLKR']

# Run the experiment from one dimension to five dimension
for algorithm in reduced_algorithm:
    for nn in range(1, 7, 1):
        # Read file LNN_Train_data.xlsx'
        org_data, org_label = LoadData.get_fnn_training_data(nn)

        # Normalize the data
        # normalized_data = preprocessing.normalize(org_data, norm='l1')
        # print(normalized_data)
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_data = min_max_scaler.fit_transform(org_data)

        # reduced_data = ra.nca(normalized_data, org_label, dim)
        # reduced_data = ra.lfda(normalized_data, org_label, dim)
        reduced_data = ra.mlkr(normalized_data, org_label, dim)

        # 呼叫不同的降維法去降維, 取特徵直
        # normalized_data = preprocessing.normalize(reduced_data, norm='l1')
        normalized_data = min_max_scaler. fit_transform(reduced_data)
        print(normalized_data)

        # Split the original data
        data1, data2 = (np.array([]) for _ in range(2))
        for element, stamp in zip(normalized_data, org_label):
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
        rel_path = '../Data/Graph/' + algorithm + '_Graph_FNN' + str(nn) + '_' + '.png'
        abs_path = os.path.join(os.path.dirname(__file__), rel_path)
        plt.savefig(abs_path)
        plt.show()
        # plt.ion()
        # plt.pause(5)
        # plt.close()
