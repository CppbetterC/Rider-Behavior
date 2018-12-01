import os
import numpy as np
import pandas as pd

from Method.LoadData import LoadData

all_label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
cluster_num = {'C1': 6, 'C2': 5, 'C3': 5, 'C4': 5, 'C5': 5, 'C6': 4}
nn_category = np.array([])
for element in all_label:
    if cluster_num[element] == 0:
        nn_category = np.append(nn_category, element)
    else:
        for num in range(cluster_num[element]):
            nn_category = np.append(nn_category, element + '_' + str(num))
print('nn_category', nn_category)

header = ['Dim' + str(i) for i in range(1, 4, 1)]
header.append('Label')
np_data = np.array([])
for nn in nn_category:
    org_data, org_label = LoadData.get_method2_fnn_train(nn)
    print(org_data.shape)
    print(org_label.shape)

    tmp = np.concatenate((org_data, org_label.reshape(-1, 1)), axis=1)
    if len(np_data) == 0:
        np_data = tmp
    else:
        np_data = np.concatenate((np_data, tmp), axis=0)

    print(tmp.shape)
    print(np_data.shape)

pd_data = pd.DataFrame(np_data, columns=header)
print(pd_data)

rel_path = '../Data/Labeling/C/method2/Test_data.xlsx'
abs_path = os.path.join(os.path.dirname(__file__), rel_path)
writer = pd.ExcelWriter(abs_path, engine='xlsxwriter')
pd_data.to_excel(writer, sheet_name='Test_Data', index=False)
writer.save()
