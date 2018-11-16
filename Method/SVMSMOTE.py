import numpy as np
import random


# The effect of this algorithm is that balance the data set'
class SVMSMOTE:

    def __init__(self, data):
        print('SVM SMOTE processing...')
        self.__org_data = data
        # self.__data1 = np.array([])
        # self.__data2 = np.array([])

        # Collect the new data and
        # Append those on the self.__org_data in the end
        # self.__expand_data = np.array([])

    def balance_data(self, upper_bound):
        length = len(self.__org_data)
        for _ in range(upper_bound - length):
            while True:
                num1 = np.random.randint(0, len(self.__org_data) - 1)
                num2 = np.random.randint(0, len(self.__org_data) - 1)
                if num1 != num2:
                    break
            self.__org_data = np.concatenate(
                [self.__org_data, [(self.__org_data[num1] + self.__org_data[num2]) / 2]])

        return self.__org_data

    # def __separate_data(self):
    #     cnt1, cnt2 = (0 for _ in range(2))
    #     data1, data2 = (np.array([]) for _ in range(2))
    #     data = self.__org_data.loc[:, ['Dim1', 'Dim2', 'Dim3']].values
    #
    #     label_data = self.__org_data.loc[:, ['label']].values
    #     for x, y in zip(data, label_data):
    #         if y == 1.0:
    #             data1 = np.append(data1, x)
    #             cnt1 += 1
    #         elif y == 0.0:
    #             data2 = np.append(data2, x)
    #             cnt2 += 1
    #         else:
    #             print('Error labeling type.')
    #             print('Please check your code.')
    #
    #     data1 = data1.reshape(-1, 3)
    #     data2 = data2.reshape(-1, 3)
    #     return data1, cnt1, data2, cnt2

    # def __cal_euclid_distance(self, all_data, choose_data, idx, close_num):
    #     dist = {}
    #     choose = []
    #     for i in range(0, len(all_data), 1):
    #         if i == idx:
    #             continue
    #         # Using the dict to record the distance between all_data[i] and choose_data
    #         dist[i] = self.__calculate(all_data[i], choose_data)
    #
    #     # Find the closed index of the self.__org_data
    #     for _ in range(close_num):
    #         try:
    #             key = min(dist, key=dist.get)
    #             choose.append(key)
    #             dist.pop(key)
    #
    #         except ValueError:
    #             pass
    #         # print('key', key)
    #     return choose

    # Calculate the sum of the label data and no_label data
    # Record the index correspond with the self.__org_data
    # def __calculate_label_sum(self, category):
    #     idx1, idx2 = [], []
    #     data1, data2 = (np.array([]) for _ in range(2))
    #     for i in range(len(self.__org_label)):
    #         if self.__org_label[i] == category:
    #             data1 = np.append(data1, self.__org_data[i])
    #             idx1.append(i)
    #         else:
    #             data2 = np.append(data2, self.__org_data[i])
    #             idx2.append(i)
    #     data1 = data1.reshape(len(idx1), -1)
    #     data2 = data2.reshape(len(idx2), -1)
    #     return data1, idx1, data2, idx2

    # @staticmethod
    # def __calculate(data1, data2):
    #     return np.linalg.norm(data1-data2)

