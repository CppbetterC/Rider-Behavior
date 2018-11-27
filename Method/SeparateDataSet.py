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
import matplotlib.pyplot as plt

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
    def show(num, data):
        kmeans = KMeans(n_clusters=num, random_state=0).fit(data)
        cluster_label = kmeans.labels_
        print('kmeans', kmeans)
        print('cluster_label', cluster_label)
        print('cluster_length', set(cluster_label))

        array_dict = {}
        for i in range(len(set(cluster_label))):
            array = np.array([])
            for j in range(len(cluster_label)):
                if cluster_label[j] == i:
                     if bool(array) == 0:
                         array = data[j]
                     else:
                        array = np.concatenate((array, data[j]), axis=1)
            array_dict[i] = array
        print('array_dict', array_dict)
        print('stop')
        xx=input()

        # plt.figure(figsize=(8, 6), dpi=100)
        # plt.title("Evaluated_Scores vs Cluster")
        # plt.xlabel('Cluster')
        # plt.ylabel('Evaluated_Scores')
        # plt.xlim(np.min(np_cluster) - 1, np.max(np_cluster) + 1)
        # plt.ylim(np.min(evaluated_scores), np.max(evaluated_scores))
        # for i in range(len(array_dict)):
        #
        #     plt.plot(, evaluated_scores, marker='o')
        # plt.savefig()
        # # plt.show("Evaluated_Scores_fnn" + str(i) + "_data.png")
        # # plt.ion()
        # # plt.pause(5)
        # plt.close()


"""
Basic parameter
"""
algorithm = ["PCA"]
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
print('<---Please Choose--->')
method = input()

for i in range(1, 7, 1):
    # Load data and reduced the dimension
    org_data, org_label = LoadData.get_fnn_training_data(i)

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
        if method == 1:
            values = SeparateDataSet.calinski_harabaz_score(j, normalized_data)

        # Use the cluster deviation to evaluate
        # 找最大的那個標準差是會最小的
        elif method == 2:
            values = SeparateDataSet.calculate_by_deviation(j, normalized_data)

        elif method == 3:
            values = SeparateDataSet.silhouette_score(j, normalized_data)

        # Use the silhouette_score to evaluate
        elif method == 4:
            SeparateDataSet.show(j, normalized_data)
            continue

        else:
            print("<---Error evaluated method--->")

        evaluated_scores = np.append(evaluated_scores, values)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.title("Evaluated_Scores vs Cluster")
    plt.xlabel('Cluster')
    plt.ylabel('Evaluated_Scores')
    plt.xlim(np.min(np_cluster) - 1, np.max(np_cluster) + 1)
    plt.ylim(np.min(evaluated_scores), np.max(evaluated_scores))
    plt.plot(np_cluster, evaluated_scores, marker='o')
    plt.savefig()
    # plt.show("Evaluated_Scores_fnn" + str(i) + "_data.png")
    # plt.ion()
    # plt.pause(5)
    plt.close()
