"""
根據ObserveData.py的結果，
Evaluated_Scores_fnn1_data.png
據觀察將 C1~C6 的每個OriginalData可以再細分成多少個類
例子: Split_C1Original_data ->
        1. Refactor_C1_1_Original_data
        2. Refactor_C1_2_Original_data

C1 -> C1_1, C2_2
C2 -> C2_1, C2_2, C2_3
C3 -> C3_1, C3_2
C4 -> C4_1, C4_2, C4_3, C4_4
C5 -> C5_1, C5_2, C5_3
C6 -> C6_1, C6_2, C6_3
"""

import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from Method.LoadData import LoadData
from Method.Normalize import Normalize
from Method.ReducedAlgorithm import ReducedAlgorithm as ra

# Dimension Reduced algorithm is tSNE
# 這邊可以調整C1~C6每個要再細分成幾類
# cluster_num = {'C1': 6, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5, 'C6': 4}
# cluster_num = {'C1': 0, 'C2': 2, 'C3': 2, 'C4': 2, 'C5': 0, 'C6': 0}
cluster_num = {'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2, 'C5': 2, 'C6': 2}
dim = 3


for key, values in cluster_num.items():
    if values == 0:
        continue
    # Load data and reduced the dimension
    print('<---' + key + ' Start--->')
    org_data, org_label = LoadData.get_split_original_data(key[1])

    # Dimension Reduced
    reduced_data = ra.tsne(org_data, dim)

    # Normalized data
    normalized_data = Normalize.normalization(reduced_data)

    kmeans = KMeans(n_clusters=values, random_state=0).fit(normalized_data)
    cluster_label = kmeans.labels_
    # print('kmeans', kmeans)
    # print('cluster_label', cluster_label)
    # print('cluster_length', set(cluster_label))
    # print(org_data)
    # print(normalized_data.shape)

    array_dict = {}
    for j in range(len(set(cluster_label))):
        array = np.array([])
        for element, label in zip(normalized_data, cluster_label):
            if label == j:
                if len(array) == 0:
                    array = element.reshape(-1, dim)
                else:
                    array = np.concatenate((array, element.reshape(-1, dim)), axis=0)
        # print('array', j, array.shape)
        array_dict[j] = array
    # print('array_dict', array_dict)
    print('<---' + key + ' Successfully--->')

    header = ['Dim' + str(i) for i in range(1, 4, 1)]
    for i in range(len(array_dict)):
        pd_data = pd.DataFrame(array_dict[i], columns=header)
        tmp_label = \
            np.array([key + '_' + str(i) for _ in range(len(array_dict[i]))]).reshape(-1, 1)
        pd_label = pd.DataFrame(tmp_label, columns=['Label'])
        result = pd.concat((pd_data, pd_label), axis=1)
        # print(result)

        # Output the result to excel
        # rel_path = '..\\Data\\Labeling\\C\\method2\\'+key+'_'+str(i)+'_Refactor_data.xlsx'
        rel_path = '..\\Data\\Labeling\\C\\method3\\'+key+'_'+str(i)+'_Refactor_data.xlsx'
        abs_path = os.path.join(os.path.dirname(__file__), rel_path)
        writer = pd.ExcelWriter(abs_path, engine='xlsxwriter')
        result.to_excel(writer, sheet_name='Labeling_Data', index=False)
        writer.save()
    print('<---Output to the excel Successfully---->')

