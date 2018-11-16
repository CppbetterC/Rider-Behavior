import numpy as np
import pandas as pd
import os

from Method.LoadData import LoadData
from Method.SVMSMOTE import SVMSMOTE


# Load the file to split as train/test data
# 選擇該類神經需要的資料集
# 例如C1, 從原始資料集抓 C1~C6
# 比重是 0.5, 0.1, 0.1, 0.1, 0.1, 0.1
# 參數數量取 500, 100, 100, 100, 100, 100
# tt 是取0.5比重的那個資料集
# t1, t2, t3, t4, t5 各取0.1比重的資料集
def build_train_data(org_data, nn_label):
    # big_num 是0.5比重的那個資料集的數量
    # small_num 是0.1比重的那個資料集的數量
    big_num, small_num = 500, 100
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


# 讀原始的資料集(264維度)
# 產生用於訓練的資料集
# 給FNN1 ~ FNN6
# org_data = LoadData.get_org_data()
# for nn in range(1, 7, 1):
#
#     data = build_train_data(org_data, nn)
#
#     path_name = './Data/Labeling/C/' + 'FNN_Train_data_' + str(nn) + '.xlsx'
#     path_name = os.path.join(os.path.dirname(__file__), path_name)
#     writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
#     data.to_excel(writer, sheet_name='Labeling_Data', index=False)
#     writer.save()


# 最後輸出一份包含所有的資料集用在 Label NN
header = ['Dim' + str(i) for i in range(1, 265, 1)]
np_data, np_label = (np.array([]) for _ in range(2))
for nn in range(1, 7, 1):
    org_data, org_label = LoadData.get_fnn_training_data(nn)
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

path_name = './Data/Labeling/C/LNN_Train_data.xlsx'
path_name = os.path.join(os.path.dirname(__file__), path_name)
writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
result.to_excel(writer, sheet_name='Labeling_Data', index=False)
writer.save()
