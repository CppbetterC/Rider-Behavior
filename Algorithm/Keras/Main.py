"""
This script implement the deep neural networks by the Keras
The model structure have one input layer, eight hidden layer, one output layer
More detail, you can print the "model.summary" to show.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.layers import SimpleRNN, LSTM ,Reshape
from keras.utils import np_utils
from numpy import genfromtxt
from keras.models import load_model
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def get_split_data():
    path_name = '../../Data/Labeling/C/Split_data.xlsx'
    path_name = os.path.join(os.path.dirname(__file__), path_name)
    excel_data = pd.read_excel(path_name)
    columns = ['AccX', 'AccY', 'AccZ',
                'GyroX', 'GyroY', 'GyroZ',
                'MagX', 'MagY', 'MagZ', 'PreX', 'PreY']
    data = excel_data.loc[:, columns].values
    labels = excel_data.loc[:, ['Label']].values.ravel()
    print('labels', labels)
    new_labels = np.array([int(e[1])-1 for e in labels])
    return data, new_labels


def get_train_data():
    path_name = '../../Data/Labeling/C/Keras_train_data.xlsx'
    path_name = os.path.join(os.path.dirname(__file__), path_name)
    excel_data = pd.read_excel(path_name)
    columns = ['AccX', 'AccY', 'AccZ',
                'GyroX', 'GyroY', 'GyroZ',
                'MagX', 'MagY', 'MagZ', 'PreX', 'PreY']
    data = excel_data.loc[:, columns].values
    labels = excel_data.loc[:, ['Label']].values.ravel()
    # print('labels', labels)
    new_labels = np.array([int(e[1])-1 for e in labels])
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


def show_train_history(train_history, train, validation, file_name):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('/Users/yurenchen/Documents/Rider-Behavior-Keras/'+file_name)
    plt.show()


# Load the data, Normalized, Train/Test data
# org_data, org_label = get_split_data()
# normalized_data = normalization(org_data)
org_data, org_label = get_train_data()

print('org_data.shape', org_data.shape)
print('org_label.shape', org_label.shape)

X_train, X_test, y_train, y_test = train_test_split(org_data, org_label, test_size=0.33, random_state=42)

y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

# Init the DNN constructor
model = Sequential()

# 這個是第一層 要設定input_shape
# units 指的是第一個 hidden layer 有幾個神經元
# 指定 input layer 有264個神經元
model.add(Dense(units=512, input_dim=11))

# 下面這些看你要幾層跟各層的數字
# 下面的是 hidden layer 的層數
# kernel_initializer='normal'
# model.add(Dense(512, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(8, activation='tanh'))
# model.add(Dense(16, activation='sigmoid'))
# model.add(Dense(16, activation='sigmoid'))

# Output layer 輸出的神經元為1個
model.add(Dense(units=6, kernel_initializer='normal', activation='softmax'))

# Compile the DNN　Structure
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])

# Show the DNN Structure
model.summary()
# print(model.summary())

# input 跟output指定好
train_history = model.fit(x=X_train, y=y_TrainOneHot, validation_split=0.2, epochs=30, batch_size=200, verbose=2)
# print('train_history.keys', train_history.history.keys())
# show_train_history(train_history, 'mean_squared_error', 'val_mean_squared_error', 'mean_squared_error.png')
# show_train_history(train_history, 'loss', 'val_loss', 'loss.png')
# print('train_history', train_history)

# 顯示訓練成果(分數)
scores = model.evaluate(X_test, y_TestOneHot)
print('scores', scores)

# 預測(prediction)
prediction = model.predict(X_test)
for x, y in zip(prediction[:10], y_TestOneHot[:10]):
    print(x, ' ', y)

# Confusion Matrix
prediction = model.predict_classes(X_test)
C_matrix = confusion_matrix(y_test, prediction)
C_accuracy = np.sum(C_matrix.diagonal())/np.sum(C_matrix)
print('C_matrix\n', C_matrix)
print('C_accuracy: ', C_accuracy)

# Get the output on each layer
inp = model.input
outputs = [layer.output for layer in model.layers]
functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions

# Testing
# np.newaxis 用在增加維度
input_shape = 11
test = np.random.random(input_shape)[np.newaxis, :]
layer_outs = [func([test]) for func in functors]
print('The output of each layer')
print('len(outputs): ', len(outputs))
print('The object of the layer: ', outputs)

for i in range(len(layer_outs)):
    print(len(layer_outs[i]))
    print('The content of the '+str(i)+' each layer: ', layer_outs[i])
    print('<--------------------------------------->')

# 模型存起來
# path_name = '/Users/yurenchen/Documents/Rider-Behavior-Keras/FinalModel.h5'
# model.save(path_name)

# json_string = model.to_json()
# with open("DNN" + str(nn) + '_Model.json', 'w') as json_file:
#     json_file.write(json_string)
# model.save_weights("DNN" + str(nn) + "_Model.h5")