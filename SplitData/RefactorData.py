import pandas as pd
import numpy as np
import random
import math
import os

from Method.Export import Export

# This script refactor the data to generate
# The equal number data set for FNN to train/test
# We need to limit the number of the data set
# Compare each excel file and pick a max  as the upper bound


class RefactorData:

    def __init__(self, l_type):
        # self.__org_data is the data
        # Load from the LDA_data file
        # self.__data1 represent the data belong l_type
        # self.__data1 represent the data not belong l_type
        # self.__count1 represent the number belong l_type
        # self.__count1 represent the number not belong l_type
        rel_path = '../Data/Labeling/C/LDA_data_' + l_type + '.xlsx'
        abs_path = os.path.join(os.path.dirname(__file__), rel_path)
        self.__org_data = pd.read_excel(abs_path)
        self.__data1, self.__count1, self.__data2, self.__count2 = self.__separate_data()

    def balance_data(self, k=3):
        if self.__count1 < self.__count2:
            for _ in range(abs(self.__count2 - self.__count1)):
                # Calculate the euclid distance with the other node
                num1 = random.randint(0, self.__count1 - 1)
                # choose_num is a list
                # we choose arbitrarily element to expand the data set
                choose_num = self.__cal_euclid_distance(self.__data1, self.__data1[num1], num1, k)
                if len(choose_num) == 0:
                    continue
                num2 = random.randint(0, len(choose_num) - 1)
                if num1 == num2:
                    continue
                self.__data1 = np.concatenate(
                    (self.__data1, np.array([(self.__data1[num1] + self.__data1[num2]) / 2])), axis=0)
        else:
            for _ in range(abs(self.__count2 - self.__count1)):
                num1 = random.randint(0, self.__count2 - 1)
                choose_num = self.__cal_euclid_distance(self.__data2, self.__data2[num1], num1, k)
                if len(choose_num) == 0:
                    continue
                num2 = random.randint(0, len(choose_num) - 1)
                if num1 == num2:
                    continue
                self.__data2 = np.concatenate(
                    (self.__data2, np.array([(self.__data2[num1] + self.__data2[num2]) / 2])), axis=0)

        label1 = np.array([1 for _ in range(len(self.__data1))])
        label2 = np.array([0 for _ in range(len(self.__data2))])

        x1 = np.concatenate((self.__data1, label1.reshape(-1, 1)), axis=1)
        x2 = np.concatenate((self.__data2, label2.reshape(-1, 1)), axis=1)
        data = np.concatenate((x1, x2), axis=0)
        return data

    def __separate_data(self):
        cnt1, cnt2 = (0 for _ in range(2))
        data1, data2 = (np.array([]) for _ in range(2))
        data = self.__org_data.loc[:, ['Dim1', 'Dim2', 'Dim3']].values

        label_data = self.__org_data.loc[:, ['label']].values
        for x, y in zip(data, label_data):
            if y == 1.0:
                data1 = np.append(data1, x)
                cnt1 += 1
            elif y == 0.0:
                data2 = np.append(data2, x)
                cnt2 += 1
            else:
                print('Error labeling type.')
                print('Please check your code.')

        data1 = data1.reshape(-1, 3)
        data2 = data2.reshape(-1, 3)
        return data1, cnt1, data2, cnt2

    def __cal_euclid_distance(self, all_data, choose_data, idx, close_num):
        dist = {}
        choose = []
        for i in range(0, len(all_data), 1):
            if i == idx:
                continue
            dist[i] = self.__calculate(all_data[i], choose_data)

        for _ in range(close_num):
            try:
                key = min(dist, key=dist.get)
                choose.append(key)
                dist.pop(key)

            except ValueError:
                pass
            # print('key', key)
        return choose

    @staticmethod
    def __calculate(data1, data2):
        # print('data1', data1)
        # print('data2', data2)
        return math.sqrt((data1[0] - data2[0]) ** 2 + (data1[1] - data2[1]) ** 2 + (data1[2] - data2[2]) ** 2)


# label_type = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
# for e in label_type:
#     refactor = RefactorData(e)
#     data_new = refactor.balance_data()
#     rel_path = '../Data/Labeling/C/Refactor_data_' + e + '.xlsx'
#     abs_path = os.path.join(os.path.dirname(__file__), rel_path)
#     Export.export_balanced_data(data_new, abs_path)


# Test git branch




