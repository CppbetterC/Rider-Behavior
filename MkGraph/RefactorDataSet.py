import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Method.LoadData import LoadData

'''
This script,
Output the 3-dimension graph
Don't refactor or generate the data
'''


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


for i in range(1, 7, 1):

    num = i
    data, label = LoadData.get_balanced_excel(num)
    norm_data = normalization(data)
    label = label.ravel()

    # print('norm_data', norm_data)
    # print(label)

    fig = plt.figure()
    ax = Axes3D(fig)

    cnt, no_cnt = 0, 0

    label_data = np.array([[]])
    no_label_data = np.array([[]])

    for element, stamp in zip(norm_data, label):
        # print('label', stamp)
        # print('x_axes', element[0])
        # print('y_axes', element[1])
        # print('z_axes', element[2])

        # print(element)

        if stamp == 1.0:
            label_data = np.append(label_data, element)

        else:
            no_label_data = np.append(no_label_data, element)

    # print(label_data)
    # print(no_label_data)
    print('len(label_data)', len(label_data))
    print('len(no_label_data)', len(no_label_data))

    new_label_data = label_data.reshape(-1, 3).T
    new_no_label_data = no_label_data.reshape(-1, 3).T

    # print(new_label_data)
    # print(new_no_label_data)

    ax.scatter(new_label_data[0], new_label_data[1], new_label_data[2], s=20)
    ax.scatter(new_no_label_data[0], new_no_label_data[1], new_no_label_data[2], s=20)

    print('The data of the C', num)
    print('X upper bound = ', max(new_label_data[0]))
    print('X lower bound = ', min(new_label_data[0]))
    print('Y upper bound = ', max(new_label_data[1]))
    print('Y lower bound = ', min(new_label_data[1]))
    print('Z upper bound = ', max(new_label_data[2]))
    print('Z lower bound = ', min(new_label_data[2]))

    # plt.legend(
    #     handles=['X upper bound', 'X lower bound', 'Y upper bound', 'Y lower bound', 'Z upper bound', 'Z lower bound'],
    #     labels=[str(max(new_label_data[0])), str(min(new_label_data[0])), str(max(new_label_data[1])),
    #            str(min(new_label_data[1])), str(max(new_label_data[2])), str(min(new_label_data[2]))], loc='best')

    plt.show()


