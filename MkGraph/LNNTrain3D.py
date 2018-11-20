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
#      'Isomap', 'HessianLLE', 'LTSA',
#      'DiffusionMaps', 'KernelPCA', 'KernelLDA', 'SNE', 'SymSNE', 'tSNE',
#      'NPE', 'LPP', 'SPE', 'LLTSA','CCA', 'MVU', 'LandmarkMVU', 'FastMVU', 'LLC',
#      'ManifoldChart', 'CFA', 'GPLVM', 'Autoencoder', 'NCA', 'MCML', 'LMNN']


"""
mds run so long
tSNE run so long

modified_lle will wrong
hessian_lle will wrong
"""
# reduced_algorithm = [
# 'lle', 'pca', 'KernelPCA', 'FactorAnalysis', 'Isomap'
# , 'ltsa_lle', 'sparse_pca', 'tSNE']

reduced_algorithm = ['Isomap']

# Run the experiment from one dimension to five dimension
for algorithm in reduced_algorithm:
    # Read file LNN_Train_data.xlsx'
    org_data, org_label = LoadData.get_lnn_training_data()

    # Normalize the data
    # normalized_data = preprocessing.normalize(org_data)
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_data = min_max_scaler.fit_transform(org_data)
    # print(normalized_data)

    # 呼叫不同的降維法去降維, 取特徵直
    # Use different reduced algorithm
    reduced_data = np.array([])
    if algorithm == 'lle':
        reduced_data = ra.lle(normalized_data, dim)

    elif algorithm == 'modified_lle':
        reduced_data = ra.modified_lle(normalized_data, dim)

    elif algorithm == 'hessian_lle':
        reduced_data = ra.hessian_lle(normalized_data, dim)

    elif algorithm == 'ltsa_lle':
        reduced_data = ra.ltsa_lle(normalized_data, dim)

    elif algorithm == 'pca':
        reduced_data = ra.pca(normalized_data, dim)

    elif algorithm == 'sparse_pca':
        reduced_data = ra.sparse_pca(normalized_data, dim)

    elif algorithm == 'mds':
        reduced_data = ra.mds(normalized_data, dim)

    elif algorithm == 'FactorAnalysis':
        reduced_data = ra.factor_analysis(normalized_data, dim)

    elif algorithm == 'Isomap':
        reduced_data = ra.isomap(normalized_data, dim)

    elif algorithm == 'KernelPCA':
        reduced_data = ra.kernel_pca(normalized_data, dim)

    elif algorithm == 'tSNE':
        reduced_data = ra.tsne(normalized_data, dim)

    else:
        print('<---Others algorithm--->')

    if not bool(reduced_data.size):
        print('<---Break Loop--->')
        break

    # normalized_data = preprocessing.normalize(reduced_data)
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_data = min_max_scaler.fit_transform(reduced_data)
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

    # Make graph
    fig = plt.figure(figsize = (8, 6), dpi = 100)
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
    rel_path = '../Data/Graph/' + algorithm + '_Graph_LNN.png'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    plt.savefig(abs_path)
    plt.ion()
    plt.pause(5)
    plt.close()
