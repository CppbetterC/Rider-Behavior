import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import confusion_matrix

from Method.LoadData import LoadData
from MkGraph.ConfusionMatrix import ConfusionMatrix
from sklearn.metrics import accuracy_score
from Algorithm.FNN import FNN

all_label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
cluster_num = \
    {all_label[0]: 6, all_label[1]: 5, all_label[2]: 5,
     all_label[3]: 5, all_label[4]: 5, all_label[5]: 4}
nn_category = {}
idx = 0
for key, value in cluster_num.items():
    if value == 0:
        nn_category[key] = idx
        idx += 1
    else:
        for num in range(value):
            nn_category[key + '_' + str(num)] = idx
            idx += 1
print('nn_category', nn_category)

fnn_label_size = 6
fnn_input_size = 3
fnn_membership_size = fnn_input_size * fnn_label_size
fnn_rule_size = 6
fnn_output_size = 1
fnn_lr = 0.001
fnn_epoch = 1
fnn_random_size = 1

fnn_threshold = 0.0


# Load the Test data
org_data, org_label = LoadData.get_method2_test()
# print('org_data.shape', org_data.shape)
# print('org_label.shape', org_label)

output_array = np.array([])

# Load the test data, forward, store
for nn in nn_category:
    print('nn -> ', nn)
    rel_path = '../Experiment/Model/FNN/' + str(nn) + '.json'
    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
    attribute = LoadData.load_fnn_weight(abs_path)
    # print(attribute)
    mean = np.asarray(attribute['Mean'])
    stddev = np.asarray(attribute['Stddev'])
    weight = np.asarray(attribute['Weight'])
    # Test the FNN
    fnn = FNN(
        fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, mean, stddev, weight, fnn_lr, 1)
    output = fnn.testing_model(org_data)
    output_array = np.append(output_array, output)

# 轉置矩陣
print(len(nn_category))
output_array = output_array.reshape(len(nn_category), -1).T
print('output_array', output_array)
print(output_array.shape)

# label encoding
y_pred = np.array([])
denominator = np.array([cluster_num[e] for e in all_label])
print('denominator', denominator)
count = [0 for _ in range(2)]
for array in output_array:
    # print('array', array)
    # print(array.shape)

    tmp = np.argwhere(array > fnn_threshold).ravel()
    # print('tmp', tmp)

    if len(tmp) == 1:
        count[0] += 1
    else:
        count[1] += 1

    molecule = np.array([0 for _ in range(6)])
    for e in tmp:
        key = [int(x[1:2]) for x, y in nn_category.items() if y == e]
        # print('key', key, type(key))
        tt = key[0]
        molecule[tt - 1] += 1
    # print('molecule', molecule)

    vector = molecule / denominator
    # print('vector', vector)

    idx = vector.argmax()+1
    # print('idx', idx)

    y_pred = np.append(y_pred, idx)


# 全部訓練資料的 confusion matrix
# Hot map
rel_path = '../Experiment/Graph/test/CM_FinalModel.png'
abs_path = os.path.join(os.path.dirname(__file__), rel_path)
y_test = [int(label[1:2]) for label in org_label]
# print(y_test)
# print(y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
print('模型準確率 -> ', accuracy_score(y_test, y_pred))
ConfusionMatrix.plot_confusion_matrix(cnf_matrix, abs_path,
                                      classes=list(set(y_test)), title='Final Model Confusion matrix')

# 找出猜錯的資料集
error_input, correct_input = (np.array([]) for _ in range(2))
for x, y, z in zip(y_test, y_pred, org_data):
    if x != y:
        if len(error_input) == 0:
            error_input = z.reshape(-1, 3)
        else:
            error_input = np.concatenate((error_input, z.reshape(-1, 3)), axis=0)
    else:
        if len(correct_input) == 0:
            correct_input = z.reshape(-1, 3)
        else:
            correct_input = np.concatenate((correct_input, z.reshape(-1, 3)), axis=0)

print('len(correct_input)', len(correct_input))
print('len(error_input)', len(error_input))

# Scatter
rel_path = '../Experiment/Graph/test/Scatter Final Model.png'
abs_path = os.path.join(os.path.dirname(__file__), rel_path)
correct_data = correct_input.T
error_data = error_input.T
fig = plt.figure(figsize=(8, 6), dpi=100)
ax = Axes3D(fig)
ax.scatter(correct_data[0], correct_data[1], correct_data[2], color='b', label='Correct data')
ax.scatter(error_data[0], error_data[1], error_data[2], color='r', label='Error data')
ax.set_title('Scatter Final Model')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(loc='lower left')
plt.savefig(abs_path)
# plt.show()
plt.ion()
plt.pause(3)
plt.close()

# Bar
rel_path = '../Experiment/Graph/test/Bar Final Model.png'
abs_path = os.path.join(os.path.dirname(__file__), rel_path)
# x_axis = ['Correct data length', 'Error data length']
# y_axis = [len(correct_input), len(error_input)]

print('Only One', count[0])
print('Others', count[1])
x_axis = ['Only One', 'Others']
y_axis = [count[0], count[1]]
plt.title('Bar Final Model')
plt.bar(x_axis, y_axis)
plt.savefig(abs_path)
# plt.show()
plt.ion()
plt.pause(3)
plt.close()

