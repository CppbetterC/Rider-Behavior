import numpy as np
import matplotlib.pyplot as plt

from Method.Normalize import Normalize
from Method.LoadData import LoadData


org_data, org_label = LoadData.get_split_data()

# org_data = Normalize.normalization(org_data)

print(org_data.shape)
print(org_label.shape)

title = 'Acc'
plt.figure(figsize=(8, 6), dpi=120)
# plt.title(title)
# plt.xlabel('Time')
# plt.ylabel('Values')

# plt.xlim(np.min(x_data) - 1, np.max(x_data) + 1)
# plt.ylim(0, 1)

# 0 -> Acc_X
# 1 -> Acc_y
# 2 -> Acc_z
x_data = np.array([i for i in range(len(org_label))])[0:500]
color_array = ['b', 'r', 'g']
label_array = ['x', 'y', 'z']
for i in range(1):
    y_data = org_data.T[i][0:500]
    plt.plot(x_data, y_data, marker='.', color=color_array[i], label=label_array[i])
plt.legend()
plt.show()
