"""
Method1
# Load the file to split as train/test data
# 選擇該類神經需要的資料集
# 例如C1, 從原始資料集抓 C1~C6
# 比重是 0.5, 0.1, 0.1, 0.1, 0.1, 0.1
# 參數數量取 500, 100, 100, 100, 100, 100
# tt 是取0.5比重的那個資料集
# t1, t2, t3, t4, t5 各取0.1比重的資料集

Method 2
# Load the file from Refactor/Refactor_C1_1.xlsx
# 目前的種類有
C1_0 255, C1_1 543, C1_2 263, C1_3 459, C1_4 343, C1_5 231
C2_0 1327, C2_1 946, C2_2 707, C2_3 930, C2_4 1137, C2_5 1322, C2_6 1197
C3_0 414, C3_1 312, C3_2 205, C3_3 320, C3_4 173, C3_5 348, C3_6 285
C4_0 410, C4_1 157, C4_2 142, C4_3 429
C5_0 70, C5_1 82, C5_2 72
C6   45
"""

import numpy as np
import pandas as pd
import os

from Method.LoadData import LoadData
from Method.SVMSMOTE import SVMSMOTE
from Method.Normalize import Normalize
from Method.ReducedAlgorithm import ReducedAlgorithm as ra


def build_from_original(org_data, nn_label):
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


def build_from_refactor(org_data, nn_label, nn_category):
    big_num, small_num = 1000, 40
    header = ['Dim' + str(i) for i in range(1, 4, 1)]
    tmp = org_data.loc[org_data['Label'] == nn_label, header].values.astype('float64')
    if len(tmp) < big_num:
        svm = SVMSMOTE(tmp)
        tmp = svm.balance_data(big_num)
    else:
        while len(tmp) > big_num:
            idx = np.random.randint(0, len(tmp)-1)
            tmp = np.delete(tmp, idx, 0)

    pd_data = pd.DataFrame(tmp, columns=header)
    pd_label = pd.DataFrame(np.array([nn_label for _ in range(len(tmp))]), columns=['Label'])
    result = pd.concat([pd_data, pd_label], axis=1)

    for label in list(set(nn_category) - {nn_label}):
        # print("label is ", label)
        tmp = org_data.loc[org_data['Label'] == label, header].values.astype('float64')
        if len(tmp) < small_num:
            svm = SVMSMOTE(tmp)
            tmp = svm.balance_data(small_num)
        else:
            while len(tmp) > small_num:
                idx = np.random.randint(0, len(tmp) - 1)
                tmp = np.delete(tmp, idx, 0)

        pd_data = pd.DataFrame(tmp, columns=header)
        pd_label = pd.DataFrame(np.array([label for _ in range(len(tmp))]), columns=['Label'])
        tmp_result = pd.concat([pd_data, pd_label], axis=1)
        result = pd.concat([result, tmp_result], axis=0)
    print('--------------------------------------')
    print('result', result)
    return result


#####################################################
print('<---Choose how to bulid trian data--->')
print('1. Build by OriginalData.xlsx')
print('2. Bulid by RefactorData.xlsx')
method = input('<---Please Choose--->: ')

if method == '1':
    # 讀原始的資料集(264維度)
    # 產生用於訓練的資料集
    # 給FNN1 ~ FNN6
    org_data = LoadData.get_org_data()
    print(org_data)
    for nn in range(1, 7, 1):
        data = build_from_original(org_data, nn)
        path_name = '../Data/Labeling/C/' + 'FNN_Train_data_' + str(nn) + '.xlsx'
        path_name = os.path.join(os.path.dirname(__file__), path_name)
        writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
        data.to_excel(writer, sheet_name='Labeling_Data', index=False)
        writer.save()

    # 最後輸出一份包含所有的資料集用在 Label NN
    header = ['Dim' + str(i) for i in range(1, 265, 1)]
    np_data, np_label = (np.array([]) for _ in range(2))
    for nn in range(1, 7, 1):
        org_data, org_label = LoadData.get_method1_fnn_train(nn)
        if nn == 1:
            np_data = org_data
            np_label = org_label
        else:
            np_data = np.concatenate([np_data, org_data], axis=0)
            np_label = np.concatenate([np_label, org_label], axis=0)

    pd_data = pd.DataFrame(np_data, columns=header)
    pd_label = pd.DataFrame(np_label, columns=['Label'])
    result = pd.concat([pd_data, pd_label], axis=1)
    print(result)

    path_name = '../Data/Labeling/C/LNN_Train_data.xlsx'
    path_name = os.path.join(os.path.dirname(__file__), path_name)
    writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
    result.to_excel(writer, sheet_name='Labeling_Data', index=False)
    writer.save()

elif method == '2':
    dim = 3
    all_label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    cluster_num = {'C1': 6, 'C2': 7, 'C3': 7, 'C4': 4, 'C5': 3, 'C6': 0}

    header = ['Dim1', 'Dim2', 'Dim3', 'Label']
    pd_data = pd.DataFrame()
    np_data = np.array([])
    nn_category = np.array([])

    for element in all_label:
        if cluster_num[element] == 0:
            nn_category = np.append(nn_category, element)
            # 讀檔, 降維, 正規化
            org_data, org_label = LoadData.get_split_original_data(element[1])
            org_label = np.array(['C'+str(i) for i in org_label])
            reduced_data = ra.tsne(org_data, dim)
            normalized_data = Normalize.normalization(reduced_data).astype('float64')
            np_tmp = np.concatenate((normalized_data, org_label.reshape(-1, 1)), axis=1)
            pd_tmp = pd.DataFrame(np_tmp, columns=header)
            pd_data = pd.concat((pd_data, pd_tmp), axis=0)

        else:
            for num in range(cluster_num[element]):
                nn_category = np.append(nn_category, element+'_'+str(num))
                normalized_data, org_label = LoadData.get_refactor_data(element, num)
                # print(normalized_data, normalized_data.shape)
                # print(org_label, org_label.shape)
                # print(org_label.shape)
                np_tmp = np.concatenate((normalized_data, org_label.reshape(-1, 1)), axis=1)
                pd_tmp = pd.DataFrame(np_tmp, columns=header)
                pd_data = pd.concat((pd_data, pd_tmp), axis=0)
    print(pd_data)

    print('nn_category', nn_category)
    for nn in nn_category:
        data = build_from_refactor(pd_data, nn, nn_category)
        path_name = '../Data/Labeling/C/method2/' + 'FNN_Train_data_' + str(nn) + '.xlsx'
        path_name = os.path.join(os.path.dirname(__file__), path_name)
        writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
        data.to_excel(writer, sheet_name='Labeling_Data', index=False)
        writer.save()

else:
    print('<---Error choose--->')
