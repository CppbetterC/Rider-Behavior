import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.layers import SimpleRNN, LSTM ,Reshape
from keras.utils import np_utils
from numpy import genfromtxt


class DNN:

    @staticmethod
    def dnn_train(x_Train, y_Train, x_Test, y_Test):
        # hidden_layer_unit = [1024, 512, 256, 128, 64, 32, 16, 16, 16, 16, 1]

        # Init the DNN constructor
        model = Sequential()

        # 這個是第一層 要設定input_shape
        # units 指的是第一個 hidden layer 有幾個神經元
        # 指定 input layer 有264個神經元
        model.add(Dense(units=1024, input_shape=(264, 1)))

        # 下面這些看你要幾層跟各層的數字
        # 下面的是 hidden layer 的層數
        model.add(Dense(512, activation='sigmoid', kernel_initializer='normal'))
        model.add(Dense(256, activation='sigmoid', kernel_initializer='normal'))
        model.add(Dense(128, activation='sigmoid', kernel_initializer='normal'))
        model.add(Dense(64, activation='sigmoid', kernel_initializer='normal'))
        model.add(Dense(32, activation='sigmoid', kernel_initializer='normal'))
        model.add(Dense(16, activation='sigmoid', kernel_initializer='normal'))
        model.add(Dense(16, activation='sigmoid', kernel_initializer='normal'))
        model.add(Dense(16, activation='sigmoid', kernel_initializer='normal'))

        # Output layer 輸出的神經元為1個
        model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))

        # Compile the DNN　Structure
        model.compile(loss='mean_squared_error',
                      optimizer='sgd',
                      metrics=['mse'])

        # Show the DNN Structure
        model.summary()
        print(model.summary())

        # input 跟output指定好
        train_history = model.fit(x=x_Train, y=y_Train, epochs=10)
        print(train_history)

        # 顯示訓練成果(分數)
        scores = model.evaluate(x_Test, y_Test)
        print(scores)

        # 預測(prediction)
        predictions = model.predict_classes(x_Test)
        print(predictions)

        # 模型存起來
        model.save('dense20.h5')
