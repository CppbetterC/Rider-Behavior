import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing

from Method.LoadData import LoadData
from Method.Normalize import Normalize
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

# Show Method List
print('<---1. tSNE--->')
print('<---2. PCA--->')
print('<---3. Isomap--->')
reduced_algorithm = input('<---Please Choose the dimension reduced algorithm--->: ')

# Best -> tSNE
# reduced_algorithm = ['tSNE']
dimension = 3

# Run the experiment from one dimension to five dimension
for algorithm in reduced_algorithm:
    for nn in range(1, 7, 1):
        # Read file LNN_Train_data.xlsx'
        org_data, org_label = LoadData.get_method1_fnn_train(nn)

        reduced_data = np.array([])
        # Dimension Reduce
        if algorithm == "NCA":
            reduced_data = ra.nca(org_data, org_label, dimension)
        elif algorithm == "tSNE" or "1":
            reduced_data = ra.tsne(org_data, dimension)
        elif algorithm == "LLE":
            reduced_data = ra.lle(org_data, dimension)
        elif algorithm == "sparse_pca":
            reduced_data = ra.sparse_pca(org_data, dimension)
        elif algorithm == "LFDA":
            reduced_data = ra.lfda(org_data, org_label, dimension)
        elif algorithm == "PCA" or "2":
            reduced_data = ra.pca(org_data, dimension)
        elif algorithm == "Isomap" or "3":
            reduced_data = ra.isomap(org_data, dimension)
        elif algorithm == 'MDS':
            reduced_data = ra.mds(org_data, dimension)
        elif algorithm == "FactorAnalysis":
            reduced_data = ra.factor_analysis(org_data, dimension)
        else:
            print("<---Not Choose the dimension reduce algorithm--->")

        # Normalize the data
        normalized_data = Normalize.normalization(reduced_data)
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
        rel_path = '../Experiment/Graph/' + algorithm + '_Graph_FNN' + str(nn) + '_' + '.png'
        abs_path = os.path.join(os.path.dirname(__file__), rel_path)
        plt.savefig(abs_path)
        plt.show()
        # plt.ion()
        # plt.pause(5)
        # plt.close()
