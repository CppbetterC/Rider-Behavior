import os, sys
import time
import copy
import datetime
import numpy as np
import pandas as pd

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn import preprocessing

from Method.LoadData import LoadData
from Algorithm.FNN import FNN
from Algorithm.LabelNN import LabelNN

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras import optimizers

"""
# Fuzzy Neural Networks Structure
# Fuzzy Data Set are 1000
"""
fnn_label_size = 6
fnn_input_size = 3
fnn_membership_size = fnn_input_size * fnn_label_size
fnn_rule_size = 6
fnn_output_size = 1
fnn_lr = 0.0001
fnn_threshold = 0.0
fnn_epoch = 10
fnn_random_size = 100
"""
Mean => [array([ 0.50669473, -0.46570734,  0.53037757, -0.72254257,  0.17860201,
       -0.12764338,  0.81339559,  0.59684539,  0.25974405,  0.75767045,
        0.17804594, -0.21298485,  0.73556324, -0.29520707,  0.86803147,
       -0.75661551, -0.65110976, -0.9745701 ]), array([ 0.182394  ,  0.01260103,  0.46563953, -0.76294295, -0.35043964,
        0.783418  ,  0.13419479, -0.10837205,  0.81555653, -0.98748479,
       -0.89320869,  0.16043943,  0.62141563, -0.64744008, -0.79666044,
       -0.22821529,  0.87576161,  0.57339858]), array([-0.10937694, -0.40237015,  0.89383654, -0.54355602,  0.03369584,
        0.67732154, -0.28860352,  0.20802387, -0.26042579,  0.68072298,
       -0.56679213,  0.07058407,  0.88769616, -0.17108611, -0.6966475 ,
        0.86178537,  0.67789535,  0.03391492]), array([ 0.23957512, -0.64101811, -0.95818916, -0.11899413,  0.7314959 ,
       -0.67209176, -0.30425164, -0.93140883,  0.57076941, -0.89718084,
       -0.11886076,  0.54976834, -0.16535632,  0.35101118, -0.97442761,
        0.65603649, -0.72904796,  0.71730959]), array([-0.39344604,  0.54134295, -0.39993413, -0.48470393, -0.54830624,
       -0.46102112,  0.423949  ,  0.70153401, -0.46260541, -0.53135179,
       -0.92396524,  0.52814762, -0.36000329,  0.94512612, -0.26154818,
        0.93190014, -0.18486666,  0.7026829 ]), array([-0.12355245,  0.73459092,  0.70344268, -0.71207231,  0.30969088,
        0.55109939,  0.09987097,  0.58031545, -0.33685509, -0.40011107,
       -0.46252946,  0.84110563, -0.74516706,  0.28888811,  0.33098466,
       -0.07529823,  0.79366295, -0.75614806])]
Stddev => [array([0.31155073, 0.36973187, 0.67493601, 0.34780312, 0.75357624,
       0.1       , 0.80806302, 0.46831281, 0.37117947, 0.1       ,
       0.88717372, 0.1       , 0.26307821, 0.85300286, 0.8590673 ,
       0.1       , 0.7970935 , 0.86788923]), array([0.45532633, 0.42588455, 0.39161196, 0.32773499, 0.97671461,
       0.30905738, 0.69003507, 0.87139127, 0.64733886, 0.42004504,
       0.97089047, 0.46059769, 0.19854513, 0.45653535, 0.17769022,
       0.34720499, 0.46404473, 0.28713708]), array([0.893226  , 0.56268322, 0.83625659, 0.20319277, 0.1348945 ,
       0.20770001, 0.56446855, 0.96202399, 0.13606116, 0.61209419,
       0.47933716, 0.14671093, 0.43341592, 0.95962147, 0.14119698,
       0.99002073, 0.82233979, 0.42579515]), array([0.26314462, 0.1       , 0.67007365, 0.93065471, 0.20057252,
       0.86372844, 0.8811297 , 0.41442071, 0.98903811, 0.963845  ,
       0.81502191, 0.32865587, 0.46581019, 0.1       , 0.1       ,
       0.41768374, 0.17139524, 0.68932579]), array([0.62833661, 0.68086214, 0.47812425, 0.80301071, 0.69577051,
       0.85726884, 0.10104335, 0.67763656, 0.99118831, 0.32982001,
       0.20685575, 0.11418727, 0.61751506, 0.57474404, 0.13784638,
       0.8537726 , 0.67164073, 0.32833513]), array([0.16155241, 0.90214686, 0.56911404, 0.24837194, 0.87766229,
       0.71873307, 0.79812542, 0.20883086, 0.68696216, 0.34034742,
       0.61147411, 0.42180384, 0.74037481, 0.48103741, 0.5135881 ,
       0.1       , 0.6211397 , 0.39591952])]
Weight => [array([-0.81239219,  0.20316367,  0.01561678, -0.01124408,  0.33901506,
        0.70596015]), array([ 0.80241194,  0.89755947,  0.72389059, -0.73059846,  0.18982407,
       -0.23326993]), array([ 0.22590401, -0.54500988,  0.25472089, -0.4346272 ,  0.34061346,
        0.54575263]), array([ 0.83239237,  0.2800558 , -0.87886474,  0.16331111,  0.82548988,
       -0.19259006]), array([ 0.03173022, -0.5291166 , -0.45034091,  0.01552542, -0.94512197,
        0.68864743]), array([ 0.62288561,  0.10825926, -0.66016099, -0.60022513, -0.32416925,
       -0.08758453])]
"""
"""
Dimension reduce algorithm
"""
# dimension_reduce_algorithm = ['LLE', 'PCA', 'Isomap']
dimension_reduce_algorithm = ['Isomap']


# Using the LLE(Locally Linear Embedding)
# To reduce the dimension
def reduce_dimension(data, algorithm_name):
    dim = fnn_input_size
    data_new = np.array([])

    if algorithm_name == 'LLE' or 'lle':
        # LLE
        embedding = LocallyLinearEmbedding(n_components=dim, eigen_solver='dense')
        data_new = embedding.fit_transform(data)

    elif algorithm_name == 'PCA' or 'pca':
        # PCA
        pca = PCA(n_components=dim)
        pca.fit(data)
        data_new = pca.transform(data)

    elif algorithm_name == 'Isomap' or 'isomap':
        # Isomap
        embedding = Isomap(n_components=dim)
        data_new = embedding.fit_transform(data)
    else:
        print('None dimension reduced')

    return data_new


# def get_fnn_output(data, m):
#     forward_output_list = np.array([])
#     for i in range(0, 6, 1):
#         # If nn == 2, 3, 4
#         # Used the DNN Model
#         if 1 <= i <= 3:
#             # load dnn model
#             # 'DNN' + str(nn) + '_Model.h5'
#             model = load_model('DNN' + str(i+1) + '_Model.h5')
#             result =
#             forward_output_list = np.append(forward_output_list, result)
#         else:
#             # print('<--- Print the FNN ' + str(nn) + ' Output--->')
#             mean = fnn_attribute['Mean'][i]
#             stddev = fnn_attribute['Stddev'][i]
#             weight = fnn_attribute['Weight'][i]
#             fnn = FNN(
#                 fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, mean, stddev, weight, fnn_lr, 1)
#             forward_output = fnn.forward(data)
#             forward_output_list = np.append(forward_output_list, forward_output)
#     return forward_output_list


"""
Test all model
"""


def test_all_model(algorithm):
    # Load file, Original_data.xlsx
    org_data, org_label = LoadData.get_test_data()

    # Reduce dimension and generate train/test data


    # normalized_data = preprocessing.normalize(reduced_data)
    # reduced_data = normalization(reduced_data)

    X_train, X_test, y_train, y_test = train_test_split(org_data, org_label, test_size=0.3)

    min_max_scaler = preprocessing.MinMaxScaler()

    X_train_new = min_max_scaler.fit_transform(X_train)
    print('X_train_new', X_train_new.shape)

    reduced_data = reduce_dimension(X_train, algorithm)
    normalized_data = min_max_scaler.fit_transform(reduced_data)
    print('normalized_data', normalized_data.shape)

    print('<---Test the Label NN Start--->')

    # 改成輸出mean, stddev, weight 來使用 fnn
    #
    fnn1_attribute = \
        {'Mean': np.array([ 0.50669473, -0.46570734,  0.53037757, -0.72254257,  0.17860201,
       -0.12764338,  0.81339559,  0.59684539,  0.25974405,  0.75767045,
        0.17804594, -0.21298485,  0.73556324, -0.29520707,  0.86803147,
       -0.75661551, -0.65110976, -0.9745701 ]),
        'Stddev': np.array([0.31155073, 0.36973187, 0.67493601, 0.34780312, 0.75357624,
       0.1, 0.80806302, 0.46831281, 0.37117947, 0.1,
       0.88717372, 0.1, 0.26307821, 0.85300286, 0.8590673,
       0.1, 0.7970935 , 0.86788923]),
        'Weight': np.array([-0.81239219,  0.20316367,  0.01561678, -0.01124408,  0.33901506,
        0.70596015])}

    fnn5_attribute = \
        {'Mean': np.array([-0.39344604,  0.54134295, -0.39993413, -0.48470393, -0.54830624,
       -0.46102112,  0.423949  ,  0.70153401, -0.46260541, -0.53135179,
       -0.92396524,  0.52814762, -0.36000329,  0.94512612, -0.26154818,
        0.93190014, -0.18486666,  0.7026829 ]),
        'Stddev': np.array([0.62833661, 0.68086214, 0.47812425, 0.80301071, 0.69577051,
       0.85726884, 0.10104335, 0.67763656, 0.99118831, 0.32982001,
       0.20685575, 0.11418727, 0.61751506, 0.57474404, 0.13784638,
       0.8537726 , 0.67164073, 0.32833513]),
        'Weight': np.array([ 0.03173022, -0.5291166 , -0.45034091,  0.01552542, -0.94512197,
        0.68864743])}

    fnn6_attribute = \
        {'Mean': np.array([-0.12355245,  0.73459092,  0.70344268, -0.71207231,  0.30969088,
        0.55109939,  0.09987097,  0.58031545, -0.33685509, -0.40011107,
       -0.46252946,  0.84110563, -0.74516706,  0.28888811,  0.33098466,
       -0.07529823,  0.79366295, -0.75614806]),
       'Stddev': np.array([0.16155241, 0.90214686, 0.56911404, 0.24837194, 0.87766229,
       0.71873307, 0.79812542, 0.20883086, 0.68696216, 0.34034742,
       0.61147411, 0.42180384, 0.74037481, 0.48103741, 0.5135881 ,
       0.1       , 0.6211397 , 0.39591952]),
        'Weight': np.array([ 0.62288561,  0.10825926, -0.66016099, -0.60022513, -0.32416925,
       -0.08758453])}

    output_list = np.array([])
    for i in range(1, 7, 1):
        if i == 1:
            mean = fnn1_attribute['Mean']
            stddev = fnn1_attribute['Stddev']
            weight = fnn1_attribute['Weight']
            fnn = FNN(
                fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, mean, stddev, weight, fnn_lr, 1)
            test_output = fnn.testing_model(normalized_data)
            output_list = np.append(output_list, test_output)

        elif i == 2:
            model = load_model('DNN2_Model.h5')
            model.summary()
            # adam = optimizers.Adam(lr=0.001)
            # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mse'])
            prediction = model.predict(X_train_new)
            predict = np.array([np.max(element) for element in prediction])
            output_list = np.append(output_list, predict)

        elif i == 3:
            model = load_model('DNN3_Model.h5')
            # adam = optimizers.Adam(lr=0.001)
            # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mse'])
            prediction = model.predict(X_train_new)
            predict = np.array([np.max(element) for element in prediction])
            output_list = np.append(output_list, predict)

        elif i == 4:
            model = load_model('DNN4_Model.h5')
            # adam = optimizers.Adam(lr=0.001)
            # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mse'])
            prediction = model.predict(X_train_new)
            predict = np.array([np.max(element) for element in prediction])
            output_list = np.append(output_list, predict)

        elif i == 5:
            mean = fnn5_attribute['Mean']
            stddev = fnn5_attribute['Stddev']
            weight = fnn5_attribute['Weight']
            fnn = FNN(
                fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, mean, stddev, weight, fnn_lr, 1)
            test_output = fnn.testing_model(normalized_data)
            output_list = np.append(output_list, test_output)

        elif i == 6:
            mean = fnn6_attribute['Mean']
            stddev = fnn6_attribute['Stddev']
            weight = fnn6_attribute['Weight']
            fnn = FNN(
                fnn_input_size, fnn_membership_size, fnn_rule_size, fnn_output_size, mean, stddev, weight, fnn_lr, 1)
            test_output = fnn.testing_model(normalized_data)
            output_list = np.append(output_list, test_output)

        else:
            print('Error label')
    # Result 挑最大的值得當該類標籤

    output_list = output_list.reshape(-1, 6)

    for x, y in zip(output_list, y_train):
        print(x, ' ', y)

    label_pred = LabelNN.label_encode(output_list)
    C_matrix = confusion_matrix(y_train, label_pred)
    C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)

    print('This is the confusion matrix(test_all_model)\n', C_matrix)
    # print(C_matrix)
    # print(C_accuracy)

    print('準確率:', accuracy_score(y_train, label_pred))

    print('<---Test the Label NN Successfully--->')
    print('<----------------------------------------------->')
    return C_accuracy


if __name__ == '__main__':

    algorithm = 'Isomap'
    # fnn_statistics = {}
    # lnn_statistics = {}

    # Use the LNN_Train.xlsx to test all model
    # All model contain FNN1 ~ FNN6 and LNN
    model_accuracy = test_all_model(algorithm)
    print('Model Accuracy', model_accuracy)
