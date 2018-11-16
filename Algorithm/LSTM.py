import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from Method.Export import Export
from Method.LoadData import LoadData
from Method.DataCombine import DataCombine

"""
This script use the LSTM(DNN)algorithm to predict behavior label
reduced dimension algorithm is LDA
Loaded file: Data\Labeling\C\LNN_Train_data.xlsx
"""


class LSTMClassification(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(LSTMClassification, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.hidden2output = nn.Linear(hidden_size, output_size)

    def __init_hidden(self, bathch_size):
        hidden = (torch.zeros(self.n_layers, bathch_size, self.hidden_size),
                  torch.zeros(self.n_layers, bathch_size, self.hidden_size))
        return hidden

    def forward(self, inputs):
        hidden = self.__init_hidden(1)
        output, hidden = self.lstm(inputs, hidden)
        # print('inputs\n', inputs)
        # print('inputs.size\n', inputs.size())
        # print('output\n', output)
        # print('hidden\n', hidden)
        # f_output = self.hidden2output(output.view(len(inputs), -1))
        # self.hidden2output(x)
        # x 是 output 和 hidden 的差別
        f_output = self.hidden2output(output.view(len(inputs), -1))
        return f_output


# Variable
dim = 3
all_accuracy = np.array([])

# Read file LNN_Train_data.xlsx to train/test
org_data, org_label = LoadData.get_lnn_training_data()

# Normalize the data
normalized_data = preprocessing.normalize(org_data)
# print(normalized_data)

# Use LDA algorithm to reduce the dimensions
lda = LinearDiscriminantAnalysis(n_components=dim)
lda.fit(normalized_data, org_label)
reduced_data = lda.transform(normalized_data)

normalized_data = preprocessing.normalize(reduced_data)
# print('normalized_data\n', normalized_data)

# Format the data, Convert numpy to Tensor
X_train, X_test, y_train, y_test = train_test_split(normalized_data, org_label, test_size=0.3)
train_data = torch.from_numpy(X_train).view(-1, 1, 1, dim).type(torch.FloatTensor)
train_label = torch.LongTensor([x-1 for x in y_train]).view(-1, 1)
print('train_data\n', train_data, train_data.size(), train_data.type())
print('train_label\n', train_label, train_label.size(), train_label.type())

test_data = torch.from_numpy(X_test).view(-1, 1, 1, dim).type(torch.FloatTensor)
test_label = torch.LongTensor([x-1 for x in y_test]).view(-1, 1)
print('train_data\n', test_data, test_data.size(), test_data.type())
print('train_label\n', test_label, test_label.size(), test_label.type())

# For RNN-LSTM variable
# RNN Structure
EPOCH = 20
BATCH_SIZE = 1
TIME_STEP = 1
LR = 0.0001

INPUT_SIZE = dim
HIDDEN_SIZE = 64
OUTPUT_SIZE = 6
N_LAYER = 1

LSTM = LSTMClassification(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYER)
print('RNN Architecture', LSTM)

optimizer = torch.optim.Adam(LSTM.parameters(), lr=LR)

# Pytorch's CrossEntropyLoss
# We don't use the OneHotEncoding to convert label
loss_func = torch.nn.CrossEntropyLoss(weight=None, size_average=True)

# Training RNN Model
print('<---Train the LSTM Start--->')
for epoch in range(EPOCH):
    for element, stamp in zip(train_data, train_label):
        # print('element\n', element)
        # print('stamp\n', stamp)

        # element = np.reshape(element, (1, 1, 3))
        # torch_data = torch.from_numpy(element)

        variable = Variable(element, requires_grad=True)
        # print('Variable\n', Variable)

        output = LSTM(variable)
        # print('output\n', output, output.size(), output.type())

        loss = loss_func(output, stamp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, EPOCH, loss.data[0]))

print('<---Train the LSTM Successfully--->')
print('<----------------------------------------------->')


# # Testing RNN Model
# torch.save(LSTM)
# torch.load(LSTM)
print('<---Test the LSTM Start--->')

LSTM.eval()

test_prediction = torch.LongTensor([])
for element in test_data:
    variable = Variable(element, requires_grad=False)
    output = LSTM(element)
    # print('output\n', output)

    value, idx = torch.max(output, 1)
    print(value, idx)

    # tmp = []
    #     # for k in range(len(output)):
    #     #     tmp.append(sum(output[k].data.numpy()))
    test_prediction = torch.cat((test_prediction, idx), 0)

print('Real label\n', test_label)
print('Prediction label\n', test_prediction)
print('Test Data accuracy: ', accuracy_score(test_label, test_prediction))

print('<---Test the LSTM Successfully--->')
print('<----------------------------------------------->')
