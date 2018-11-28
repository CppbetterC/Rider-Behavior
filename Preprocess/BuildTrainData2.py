"""
# Load the file from Refactor/Refactor_C1_1.xlsx
# 選擇該個類神經需要的資料集
# 例如C1, 從原始資料集抓 C1~C6
# 比重是 0.5, 0.1, 0.1, 0.1, 0.1, 0.1
# 參數數量取 500, 100, 100, 100, 100, 100
# 目前的種類有
# C1_0, C1_1,
# C2_0, C2_1, C2_2,
# C3_0, C3_1
# C4_0, C4_1, C4_2, C4_3
# C5
# C6
"""
# 'C1': 2, 'C2': 3, 'C3': 2, 'C4': 4, 'C5': 0, 'C6': 0

import numpy as np
import pandas as pd
import os

from sklearn.manifold import TSNE

from Method.LoadData import LoadData
from Method.SVMSMOTE import SVMSMOTE
from Method.Normalize import Normalize
from Method.ReducedAlgorithm import ReducedAlgorithm as ra


def build_train_data(org_data, nn_label):
    # big_num 是0.5比重的那個資料集的數量
    # small_num 是0.1比重的那個資料集的數量
    big_num, small_num = 2000, 400
    header = ['Dim' + str(i) for i in range(1, 265, 1)]
    category = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    # 找出佔比重較大的資料集(0.5)
    tt_label = 'C' + str(nn_label)
    tmp = org_data.loc[org_data['Label'] == tt_label, header].values
    # print('1', tmp)
    if len(tmp) < big_num:
        # 擴增資料集到應該的數量
        svm = SVMSMOTE(tmp)
        tmp = svm.balance_data(big_num)
        # print('2', tmp)
    else:
        while len(tmp) > big_num:
            idx = np.random.randint(0, len(tmp)-1)
            tmp = np.delete(tmp, idx, 0)
        # print('3', tmp)

    pd_data = pd.DataFrame(tmp, columns=header)
    pd_label = pd.DataFrame(np.array([tt_label for _ in range(len(tmp))]), columns=['Label'])
    result = pd.concat([pd_data, pd_label], axis=1)

    # print('r1', result.head)

    # 找出佔其他比重的資料集(0.1)
    for label in list(set(category) - {tt_label}):
        print("label is ", label)
        tmp = org_data.loc[org_data['Label'] == label, header].values
        # print('4', tmp)
        if len(tmp) < small_num:
            svm = SVMSMOTE(tmp)
            tmp = svm.balance_data(small_num)
            # print('5', tmp)
        else:
            while len(tmp) > small_num:
                idx = np.random.randint(0, len(tmp) - 1)
                tmp = np.delete(tmp, idx, 0)
            # print('6', tmp)

        pd_data = pd.DataFrame(tmp, columns=header)
        pd_label = pd.DataFrame(np.array([label for _ in range(len(tmp))]), columns=['Label'])
        tmp_result = pd.concat([pd_data, pd_label], axis=1)
        result = pd.concat([result, tmp_result], axis=0)

    # print('r2', result.head)
    print('--------------------------------------')
    return result


#######################################################
# Start

dim = 3
all_label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
cluster_num = {'C1': 2, 'C2': 3, 'C3': 2, 'C4': 4, 'C5': 0, 'C6': 0}

header = ['Dim1', 'Dim2', 'Dim3', 'Label']
pd_data = pd.DataFrame()
np_data = np.array([])

for element in all_label:
    for num in range(cluster_num[element]):
        print('element', element)
        print('num', num)
        if cluster_num[element] == 0:
            org_data, org_label = LoadData.get_split_original_data(num)
            # 降維法和正規化
            reduced_data = ra.tsne(org_data, dim)
            normalized_data = Normalize.normalization(reduced_data)

        else:
            normalized_data, org_label = LoadData.get_refactor_data(element, num)

        # print('org_data', org_data)
        # print('org_label', org_label)

        print(normalized_data.shape)
        print(org_label.shape)

        np_tmp = np.concatenate((normalized_data, org_label.reshape(-1, 1)), axis=1)
        pd_tmp = pd.DataFrame(np_tmp, columns=header)
        pd_data = pd.concat((pd_data, pd_tmp), axis=0)

print(pd_data)
