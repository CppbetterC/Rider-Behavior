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


class SeparateDataSet:

    @staticmethod
    def calinski_harabaz_score(num, data):
        kmeans = KMeans(n_clusters=num, random_state=0).fit(data)
        prediction_label = kmeans.predict(data)
        return metrics.calinski_harabaz_score(data, prediction_label)

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


"""
Basic parameter
"""
algorithm = ["PCA"]
dim = 3
method = 2

start_cluster = 2
end_cluster = 10
np_cluster = np.array([i for i in range(start_cluster, end_cluster + 1, 1)])

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

    evaluated_scores = np.array([])
    for j in range(start_cluster, end_cluster + 1, 1):
        # Use the calinski_harabaz_score to evaluate
        if method == 1:
            evaluated_scores =\
                np.append(evaluated_scores, SeparateDataSet.calinski_harabaz_score(j, reduced_data))

        # Use the cluster deviation to evaluate
        # 找最大的那個標準差是會最小的
        elif method == 2:
            evaluated_scores =\
                np.append(evaluated_scores, SeparateDataSet.calculate_by_deviation(j, reduced_data))

        else:
            print("<---Error evaluated method--->")

    plt.figure(figsize=(8, 6), dpi=80)
    plt.title("Evaluated_Scores vs Cluster")
    plt.xlabel('Cluster')
    plt.ylabel('Evaluated_Scores')
    plt.xlim(np.min(np_cluster) - 1, np.max(np_cluster) + 1)
    plt.ylim(np.min(evaluated_scores), np.max(evaluated_scores))
    plt.plot(np_cluster, evaluated_scores, marker='o')
    plt.show()
    # plt.ion()
    # plt.pause(5)
    plt.close()
