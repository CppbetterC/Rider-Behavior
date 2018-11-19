# import os, sys
# import time
# import copy
# import datetime
# import numpy as np
# import pandas as pd
#
# from sklearn.manifold import LocallyLinearEmbedding
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# from sklearn.decomposition import PCA,
# from sklearn.manifold import Isomap
# from sklearn import preprocessing
#
# from Method.LoadData import LoadData
# from Algorithm.FNN import FNN
# from Algorithm.LabelNN import LabelNN
#
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.models import load_model
#
# """
# # Fuzzy Neural Networks Structure
# # Fuzzy Data Set are 1000
# """
# fnn_label_size = 6
# fnn_input_size = 3
# fnn_membership_size = fnn_input_size * fnn_label_size
# fnn_rule_size = 6
# fnn_output_size = 1
# fnn_lr = 0.0001
# fnn_threshold = 0.0
# fnn_epoch = 10
# fnn_random_size = 100
#
#
# """
# # Label Neural Network Structure
# # Label NN Data Set are 6000
# """
# lnn_input_size = 6
# lnn_hidden_size = 6
# lnn_output_size = 6
# lnn_lr = 0.0001
# lnn_epoch = 1
# lnn_random_size = 150
#
# """
# Dimension reduce algorithm
# """
# # dimension_reduce_algorithm = ['LLE', 'PCA', 'Isomap']
# dimension_reduce_algorithm = ['Isomap']
#
#
# # Create a storage to new picture
# def makedir(path, algorithm_name):
#     # Get the timestamp
#     now = datetime.datetime.now()
#     timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
#
#     try:
#         os.mkdir(path + algorithm_name)
#         os.chdir(path + algorithm_name)
#     except FileExistsError:
#         os.chdir(path + algorithm_name)
#         print("<---FileExistsError(Main.py)--->")
#
#     # with open('README.txt', 'w', encoding='utf-8') as f:
#     #     f.writelines('timestamp ->' + timestamp)
#     #     f.writelines('algorithm_name' + algorithm_name)
#
#     # return os.getcwd()
#
#
# # Using the LLE(Locally Linear Embedding)
# # To reduce the dimension
# def reduce_dimension(data, algorithm_name):
#     dim = fnn_input_size
#     data_new = np.array([])
#
#     if algorithm_name == 'LLE' or 'lle':
#         # LLE
#         embedding = LocallyLinearEmbedding(n_components=dim, eigen_solver='dense')
#         data_new = embedding.fit_transform(data)
#
#     elif algorithm_name == 'PCA' or 'pca':
#         # PCA
#         pca = PCA(n_components=dim)
#         pca.fit(data)
#         data_new = pca.transform(data)
#
#     elif algorithm_name == 'Isomap' or 'isomap':
#         # Isomap
#         embedding = Isomap(n_components=dim)
#         data_new = embedding.fit_transform(data)
#     else:
#         print('None dimension reduced')
#
#     return data_new
#
#
# def get_fnn_output(data, fnn_attribute):
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
#
#
# """
# Test all model
# """
#
#
# def test_all_model(fnn_attribute, algorithm):
#     # Load file, Original_data.xlsx
#     org_data, org_label = LoadData.get_test_data()
#
#     # Reduce dimension and generate train/test data
#     reduced_data = reduce_dimension(org_data, algorithm)
#     normalized_data = preprocessing.normalize(reduced_data)
#     # reduced_data = normalization(reduced_data)
#
#     X_train, X_test, y_train, y_test = train_test_split(normalized_data, org_label, test_size=0.3)
#
#     print('<---Test the Label NN Start--->')
#     test_output_list = np.array([])
#
#     # 改成輸出mean, stddev, weight 來使用 fnn
#     #
#
#     for train_data, train_label in zip(X_train, y_train):
#         result = get_fnn_output(train_data, fnn_attribute)
#
#     # Result 挑最大的值得當該類標籤
#
#     test_output_list = test_output_list.reshape(-1, 6)
#     label_pred = LabelNN.label_encode(test_output_list)
#     C_matrix = confusion_matrix(y_train, label_pred)
#     C_accuracy = np.sum(C_matrix.diagonal()) / np.sum(C_matrix)
#
#     print('This is the confusion matrix(test_all_model)\n', C_matrix)
#     # print(C_matrix)
#     # print(C_accuracy)
#
#     print('<---Test the Label NN Successfully--->')
#     print('<----------------------------------------------->')
#     return C_accuracy
#
#
# if __name__ == '__main__':
#
#         algorithm = 'Isomap'
#         fnn_statistics = {}
#         lnn_statistics = {}
#
#         # Use the LNN_Train.xlsx to test all model
#         # All model contain FNN1 ~ FNN6 and LNN
#         model_accuracy = test_all_model(fnn_statistics, lnn_statistics, algorithm)
#         print('Model Accuracy', model_accuracy)
