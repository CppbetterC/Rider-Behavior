import numpy as np
import matplotlib.pyplot as plt

from Method.Normalize import Normalize
from Method.LoadData import LoadData


def smoothing(data):
    one_std_deviation = np.std(data)
    means = np.mean(data)
    up_bound = means + one_std_deviation * 3
    low_bound = means - one_std_deviation * 3
    tmp = np.array([])
    for i in range(0, len(data), 1):
        if data[i] > up_bound or data[i] < low_bound:
            print('smooth')
            if 0 < i < len(data) - 1:
                tmp = np.append(tmp, ((data[i-2]+data[i-1]+data[i+1]+data[i+2])/4))
        else:
            tmp = np.append(tmp, data[i])
    return tmp, up_bound, low_bound


feature = 0
org_data, org_label = LoadData.get_split_data()
# org_data = Normalize.normalization(org_data.T[0])

print(org_data.shape)
print(org_label.shape)

# title = 'Acc'
plt.figure(figsize=(8, 6), dpi=120)
# plt.title(title)
# plt.xlabel('Time')
# plt.ylabel('Values')

# plt.xlim(np.min(x_data) - 1, np.max(x_data) + 1)
plt.ylim(200, 1500)

# 0 -> Acc_X
# 1 -> Acc_y
# 2 -> Acc_z
x_data = np.array([i for i in range(len(org_label[:500]))])
color_array = ['b', 'r', 'g']
label_array = ['z', 'x', 'y']
cnt = 0
for i in range(0, 1, 1):
    # y_data = org_data.T[feature][:500]

    smooth_data, up_bound, low_bound = smoothing(org_data.T[feature][:500])
    y_data = smooth_data

    plt.plot(x_data, y_data, marker='.', color=color_array[cnt], label=label_array[cnt])

    # one_std_deviation = np.std(y_data)
    # means = np.mean(y_data)
    # up_bound = means + one_std_deviation * 3
    # low_bound = means - one_std_deviation * 3

    y_data = np.array([up_bound for _ in range(len(org_label[:500]))])
    plt.plot(x_data, y_data, marker='.', color='m', label='upper bound')

    y_data = np.array([low_bound for _ in range(len(org_label[:500]))])
    plt.plot(x_data, y_data, marker='.', color='red', label='lower bound')

    cnt += 1

plt.legend()
plt.show()
