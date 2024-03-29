import os
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.layers import SimpleRNN, LSTM ,Reshape
from keras.utils import np_utils
from numpy import genfromtxt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from sklearn.metrics import confusion_matrix

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


class DNN:

    @staticmethod
    def dnn_train(nn, x_Train, y_Train, x_Test, y_Test):

        # hidden_layer_unit = [1024, 512, 256, 128, 64, 32, 16, 16, 16, 16, 1]

        # Init the DNN constructor
        model = Sequential()

        # 這個是第一層 要設定input_shape
        # units 指的是第一個 hidden layer 有幾個神經元
        # 指定 input layer 有264個神經元
        model.add(Dense(units=1024, input_dim=264))

        # 下面這些看你要幾層跟各層的數字
        # 下面的是 hidden layer 的層數
        # kernel_initializer='normal'
        model.add(Dense(512, activation='tanh'))
        model.add(Dense(256, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(16, activation='tanh'))
        model.add(Dense(8, activation='tanh'))
        model.add(Dense(16, activation='sigmoid'))
        model.add(Dense(16, activation='sigmoid'))
        # model.add(Dense(16, activation='sigmoid'))
        # model.add(Dense(16, activation='sigmoid'))
        # model.add(Dense(16, activation='sigmoid'))

        # Output layer 輸出的神經元為1個
        model.add(Dense(2, activation='softmax'))

        # Compile the DNN　Structure
        adam = optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mse'])

        # Show the DNN Structure
        model.summary()
        # print(model.summary())

        # input 跟output指定好
        train_history = model.fit(x=x_Train, y=y_Train, epochs=100)
        # train_history

        # 顯示訓練成果(分數)
        scores = model.evaluate(x_Test, y_Test)
        print('scores', scores)

        # 預測(prediction)
        prediction = model.predict(x_Test)
        # print('prediction', prediction)
        for x, y in zip(prediction, y_Test):
            print(x, ' ', y)
        # print('hi', type(predictions), predictions.shape)

        # 模型存起來
        # print('hi model', os.getcwd())
        # model.save('DNN' + str(nn) + '_Model.(h5')
        json_string = model.to_json()
        with open("DNN" + str(nn) + '_Model.json', 'w') as json_file:
            json_file.write(json_string)
        model.save_weights("DNN" + str(nn) + "_Model.h5")
        print("Save the Model", str(nn), "in the disk.")
