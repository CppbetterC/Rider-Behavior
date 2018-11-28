"""
This script was used to observe the data set
If Original_data.xlsx was poor for the neural networks
We need to refactor the data set
Split the Original_data.xlsx to more detail data
It denote the data set will have more label(cluster)
We will use the standard deviation to evaluate
x-axis is the number of the cluster
y-axis is the score that were calculated by max deviation of the data

Two categories, kmeans and MiniBatchKMeans

###############################################
Evaluation algorithm -> Calinski-Harabasz Index
Calinski-Harabasz分數值ss越大則聚類效果越好
類別內部數據的共變異數(協方差)越小越好，類別之間的共變異數(協方差)越大越好
###############################################
Evaluation algorithm -> the max deviation of all cluster
去計算分成 N 群中的最大的那個標準差，
如果分成 N 群最大的標準差是比較小的，
那將會代表分成 N 群是較好的分群結果
###############################################

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import metrics

from Method.LoadData import LoadData
from Method.ReducedAlgorithm import ReducedAlgorithm as ra
from Method.Normalize import Normalize


class SeparateDataSet:

    @staticmethod
    def calinski_harabaz_score(num, data):
        kmeans = KMeans(n_clusters=num, random_state=0).fit(data)
        cluster_label = kmeans.labels_
        return metrics.calinski_harabaz_score(data, cluster_label)

    @staticmethod
    def calculate_by_deviation(num, data):
        kmeans = KMeans(n_clusters=num, random_state=0).fit(data)
        cluster_label = kmeans.labels_
        dist_data = kmeans.inertia_
        print('kmeans', kmeans)
        print('cluster_label', cluster_label)
        print('cluster_length', set(cluster_label))
        print('dis_data', dist_data)

        std = np.array([])
        for i in range(len(set(cluster_label))):
            array = np.array([])
            for j in range(len(cluster_label)):
                if cluster_label[j] == i:
                    array = np.append(array, data[j])
            std = np.append(std, np.std(array))
        return np.max(std)

    @staticmethod
    def silhouette_score(num, data):
        kmeans = KMeans(n_clusters=num, random_state=0).fit(data)
        cluster_label = kmeans.labels_
        return metrics.silhouette_score(data, cluster_label)

    @staticmethod
    def show(num, data, data_label, label_type, output):
        kmeans = KMeans(n_clusters=num, random_state=0).fit(data)
        cluster_label = kmeans.labels_
        print('kmeans', kmeans)
        print('cluster_label', cluster_label)
        print('cluster_length', set(cluster_label))

        print(data)
        print(data.shape)

        array_dict = {}
        for i in range(len(set(cluster_label))):
            array = np.array([])
            for element, label in zip(data, cluster_label):
                if label == i:
                    if len(array) == 0:
                        array = element.reshape(-1, 3)
                    else:
                        array = np.concatenate((array, element.reshape(-1, 3)), axis=0)
            array_dict[i] = array
        print('array_dict', array_dict)

        """
        Check print the result or not
        """
        if output == "y":
            header = ['Dim' + str(i) for i in range(1, 265, 1)]
            result = pd.DataFrame()
            for j in range(len(array_dict)):
                pd_data = pd.DataFrame(array_dict[j], columns=header)
                tmp_label = ['C' + str(e) + '_' + str(j) for e in data_label]
                pd_label = pd.DataFrame(np.array(tmp_label), columns='Label')
                result = pd.concat((pd_data, pd_label), axis=1)
            print(result)
            xxx=input()
            # continue

        # Make Scatter Graph
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = Axes3D(fig)
        cmap = plt.cm.get_cmap('viridis')
        ax.set_title("Data scatter with cluster_" + str(num) + '(' + label_type + ')')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        for i in range(len(array_dict)):
            tmp = array_dict[i].T
            ax.scatter(tmp[0], tmp[1], tmp[2], marker='o', cmap=cmap)
        plt.savefig("..\\Experiment\\ClusterScore\\"
                    "Data scatter with cluster_" + str(num) + '(' + label_type + ')' + ".png")
        # plt.show()
        plt.ion()
        plt.pause(3)
        plt.close()


"""
Basic parameter
"""
algorithm = ["tSNE"]
dim = 3
# method = 3

start_cluster = 2
end_cluster = 10
np_cluster = np.array([i for i in range(start_cluster, end_cluster + 1, 1)])


# Show Method List
print('<---1. Use the calinski_harabaz_score to evaluate--->')
print('<---2. Use the calculate_by_deviation to evaluate--->')
print('<---3. Use the silhouette_score to evaluate--->')
print('<---4. Show Cluster Scatter--->')
method = input('<---Please Choose--->: ')
output_excel = 'n'
if method == '4':
    output_excel = input('<---Output Excel?(y/n)--->: ')

for i in range(1, 7, 1):
    # Load data and reduced the dimension
    org_data, org_label = LoadData.get_split_original_data(i)

    reduced_data = np.array([])
    if algorithm == 'PCA' or 'pca':
        reduced_data = ra.pca(org_data, dim)
    elif algorithm == 'Isomap' or 'isomap':
        reduced_data = ra.isomap(org_data, dim)
    elif algorithm == 'tSNE':
        reduced_data = ra.tsne(org_data, dim)
    else:
        print('<---None dimension reduced--->')

    normalized_data = Normalize.normalization(reduced_data)

    evaluated_scores = np.array([])
    values = np.array([])
    for j in range(start_cluster, end_cluster + 1, 1):
        # Use the calinski_harabaz_score to evaluate
        if method == '1':
            values = SeparateDataSet.calinski_harabaz_score(j, normalized_data)

        # Use the cluster deviation to evaluate
        # 找最大的那個標準差是會最小的
        elif method == '2':
            values = SeparateDataSet.calculate_by_deviation(j, normalized_data)

        elif method == '3':
            values = SeparateDataSet.silhouette_score(j, normalized_data)

        # Use the silhouette_score to evaluate
        elif method == '4':
            SeparateDataSet.show(j, normalized_data, org_label, 'C'+str(i), output_excel)
            continue

        else:
            print("<---Error evaluated method--->")

        evaluated_scores = np.append(evaluated_scores, values)

    if not len(evaluated_scores) == 0:
        plt.figure(figsize=(8, 6), dpi=80)
        plt.title("Evaluated_Scores vs Cluster")
        plt.xlabel('Cluster')
        plt.ylabel('Evaluated_Scores')
        plt.xlim(np.min(np_cluster) - 1, np.max(np_cluster) + 1)
        plt.ylim(np.min(evaluated_scores), np.max(evaluated_scores))
        plt.plot(np_cluster, evaluated_scores, marker='o')
        plt.savefig("..\\Experiment\\ClusterScore\\Evaluated_Scores_fnn" + str(i) + "_data.png")

        plt.show()
        # plt.ion()
        # plt.pause(5)
        plt.close()
