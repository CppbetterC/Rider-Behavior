"""
根據ObserveData.py的結果，
Evaluated_Scores_fnn1_data.png
據觀察將 C1~C6 的每個OriginalData可以再細分成多少個類
例子: Split_C1Original_data ->
        1. Split_C1_1_Original_data
        2. Split_C1_2_Original_data

C1 -> C1_1, C2_2
C2 -> C2_1, C2_2, C2_3
C3 -> C3_1, C3_2
C4 -> C4_1, C4_2, C4_3, C4_4
C5 -> C5_1, C5_2, C5_3
C6 -> C6_1, C6_2, C6_3
"""

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from Method.LoadData import LoadData

cluster_num = {1: 2, 2: 3, 3: 2, 4: 4, 5: 0, 6: 0}

for i in range(1, 7, 1):
    # Load data and reduced the dimension
    org_data, org_label = LoadData.get_split_original_data(i)
    if cluster_num[i] == 0:
        continue
    kmeans = KMeans(n_clusters=cluster_num[i], random_state=0).fit(org_data)
    cluster_label = kmeans.labels_

    print('kmeans', kmeans)
    print('cluster_label', cluster_label)
    print('cluster_length', set(cluster_label))
    print(org_data)
    print(org_data.shape)

    array_dict = {}
    for j in range(len(set(cluster_label))):
        array = np.array([])
        for element, label in zip(org_data, cluster_label):
            if label == j:
                if len(array) == 0:
                    array = element.reshape(-1, 3)
                else:
                    array = np.concatenate((array, element.reshape(-1, 3)), axis=0)
        array_dict[j] = array
    print('array_dict', array_dict)
    print('<---C'+str(i)+' Successfully--->')

    header = ['Dim' + str(i) for i in range(1, 265, 1)]
    result = pd.DataFrame()
    for j in range(len(array_dict)):
        pd_data = pd.DataFrame(array_dict[j], columns=header)
        tmp_label = ['C'+str(e)+'_'+str(j) for e in org_label]
        pd_label = pd.DataFrame(np.array(tmp_label), columns='Label')
        result = pd.concat((pd_data, pd_label), axis=1)
    print(result)
    x=input()

