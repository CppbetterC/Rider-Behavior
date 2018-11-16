import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Method.LoadData import LoadData


# Normalization the data
# The interval between -1 and 1
def normalization(data):
    new_data = np.array([])
    tmp = data.T
    length = len(tmp[0])
    for array in tmp:
        sub_array = []
        max_value = max(array)
        min_value = min(array)
        for element in array:
            sub_array.append((2 * (element - min_value) / (max_value - min_value)) - 1)
        new_data = np.append(new_data, sub_array)
    new_data = new_data.reshape(-1, length).T
    return new_data

# Generate the original data
file_name = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
all_data = pd.DataFrame(columns=['Dim1', 'Dim2', 'Dim3', 'label'])

for name in file_name:
    rel_path = '../Data/Labeling/C/Refactor_data_' + name + '.xlsx'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)

    tmp_data = pd.read_excel(abs_path)

    # This index is the actually index
    # numeric from 0
    idx = tmp_data.index[tmp_data['label'] == 0].tolist()
    tmp_data = tmp_data.drop(idx)

    tmp_data['label'].replace(1, int(name[1]), inplace=True)

    all_data = pd.concat([all_data, tmp_data], ignore_index=True)

rel_path = '../Data/Labeling/C/Test_data.xlsx'
abs_path = os.path.join(os.path.dirname(__file__), rel_path)


print(all_data.shape)
print(all_data.head())

# writer = pd.ExcelWriter(abs_path)
# all_data.to_excel(writer, sheet_name='Labeling_Data', index=False)
#
# writer.save()

# Load file to print a photo (optional)
data1, data2, data3, data4, data5, data6 = (np.array([]) for _ in range(6))
x, y = LoadData.get_test_data()
X_new = normalization(x)
for element, stamp in zip(X_new, y.ravel()):
    if stamp == 1:
        data1 = np.append(data1, element)
    elif stamp == 2:
        data2 = np.append(data2, element)
    elif stamp == 3:
        data3 = np.append(data3, element)
    elif stamp == 4:
        data4 = np.append(data4, element)
    elif stamp == 5:
        data5 = np.append(data5, element)
    elif stamp == 6:
        data6 = np.append(data6, element)
    else:
        print('Error label.')
        print('Please check your code.')

    data1 = data1.reshape(-1, 3).T
    data2 = data2.reshape(-1, 3).T
    data3 = data4.reshape(-1, 3).T
    data4 = data4.reshape(-1, 3).T
    data5 = data5.reshape(-1, 3).T
    data6 = data6.reshape(-1, 3).T

fig = plt.figure()
ax = Axes3D(fig)
color = ['r', 'g', 'b']
size = 10
ax.scatter(data1[0], data1[1], data1[2], s=size)
ax.scatter(data2[0], data2[1], data2[2], s=size)
ax.scatter(data3[0], data3[1], data3[2], s=size)
ax.scatter(data4[0], data4[1], data4[2], s=size)
ax.scatter(data5[0], data5[1], data5[2], s=size)
ax.scatter(data6[0], data6[1], data6[2], s=size)
plt.show()



