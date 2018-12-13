import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def get_split_data():
    path_name = 'Split_data.xlsx'
    path_name = os.path.join(os.path.dirname(__file__), path_name)
    excel_data = pd.read_excel(path_name)
    columns = ['AccX', 'AccY', 'AccZ',
                'GyroX', 'GyroY', 'GyroZ',
                'MagX', 'MagY', 'MagZ', 'PreX', 'PreY']
    data = excel_data.loc[:, columns].values
    labels = excel_data.loc[:, ['Label']].values.ravel()
    print('labels', labels)
    new_labels = np.array([int(e[1]) for e in labels])
    return data, new_labels

def normalization(data):
    new_data = np.array([])
    tmp = data.T
    length = len(tmp[0])
    for array in tmp:
        sub_array = []
        max_value = max(array)
        min_value = min(array)
        for element in array:
            sub_array.append(2 * ((element - min_value) / (max_value - min_value)) - 1)
        new_data = np.append(new_data, sub_array)
    new_data = new_data.reshape(-1, length).T
    return new_data

def show_data_length():
    """
    C1: 2094, C2: 7566, C3: 2057, C4: 1138, C5: 224, C6: 45
    """
    path_name = 'Split_data.xlsx'
    path_name = os.path.join(os.path.dirname(__file__), path_name)
    excel_data = pd.read_excel(path_name)
    columns = ['AccX', 'AccY', 'AccZ',
                'GyroX', 'GyroY', 'GyroZ',
                'MagX', 'MagY', 'MagZ', 'PreX', 'PreY']
    data = excel_data.loc[:, columns].values
    labels = excel_data.loc[:, ['Label']]
    print(labels.loc[labels['Label']=='C5'].values.shape)

def get_specialize_data(label):
    path_name = 'Split_data.xlsx'
    path_name = os.path.join(os.path.dirname(__file__), path_name)
    excel_data = pd.read_excel(path_name)
    columns = ['AccX', 'AccY', 'AccZ',
                'GyroX', 'GyroY', 'GyroZ',
                'MagX', 'MagY', 'MagZ', 'PreX', 'PreY']
    temp = excel_data.loc[excel_data['Label']==label]
    data = temp.loc[:, columns].values
    # labels = temp.loc[:, ['Label']].values
    return data

def balance_data(data, upper_bound):
    length = len(data)
    for _ in range(upper_bound - length):
        while True:
            num1 = np.random.randint(0, len(data) - 1)
            num2 = np.random.randint(0, len(data) - 1)
            if num1 != num2:
                break
        data = np.concatenate([data, [(data[num1] + data[num2]) / 2]])
    return data

# show_data_length()
# org_data, org_label = get_split_data()
# normalized_data = normalization(org_data)

limit_num = 7000
labels = ['C'+str(i) for i in range(1, 7, 1)]
result = pd.DataFrame()
for label in labels:
    org_data = get_specialize_data(label)
    # print(type(org_data), org_data.shape)

    normalized_data = normalization(org_data)
    # print(type(normalized_data), normalized_data.shape)

    if len(normalized_data) < limit_num:
        normalized_data = balance_data(normalized_data, limit_num)
        print('normalized_data.shape', normalized_data.shape)

    else:
        while len(normalized_data) > limit_num:
            idx = np.random.randint(0, len(normalized_data)-1)
            normalized_data = np.delete(normalized_data, idx, 0)

    header = ['AccX', 'AccY', 'AccZ',
                'GyroX', 'GyroY', 'GyroZ',
                'MagX', 'MagY', 'MagZ', 'PreX', 'PreY']
    pd_data = pd.DataFrame(normalized_data, columns=header)
    pd_label = pd.DataFrame(np.array([label for _ in range(len(normalized_data))]), columns=['Label'])
    local_result = pd.concat([pd_data, pd_label], axis=1)
    if len(result) == 0:
        result = local_result
    else:
        result = pd.concat((result, local_result), axis=0)

# print(result)

import os
print(os.getcwd())
# /Users/yurenchen/Documents/Python
path_name = '/Users/yurenchen/Documents/Rider-Behavior-Keras/Train_data.xlsx'
writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
result.to_excel(writer, sheet_name='Labeling_Data', index=False)
writer.save()

