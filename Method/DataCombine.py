import numpy as np

from sklearn import preprocessing

from Method.Formula import Formula


class DataCombine:

    @staticmethod
    def combine(org_data):
        print('<---Behavior Distinguish--->')
        # print(org_data)

        min_max_scalar = preprocessing.MinMaxScaler()
        normalized_data = min_max_scalar.fit_transform(org_data.astype('float64'))
        normalized_data = normalized_data.T
        # print(normalized_data, normalized_data.shape)

        # windows2, windows4, windows6 = (np.array([]) for _ in range(3))
        windows2 = DataCombine.__moving_windows(normalized_data, 2)
        windows4 = DataCombine.__moving_windows(normalized_data, 4)
        windows6 = DataCombine.__moving_windows(normalized_data, 6)

        result = np.concatenate((windows2, windows4, windows6), axis=1)
        # print('result', result, result.shape)
        return result

    @staticmethod
    def __moving_windows(data, windows_size):
        result = np.array([])
        for i in range(len(data)):
            feature = np.array([])
            for j in range(len(data[i])):
                windows_data = np.array([]).astype('float64')
                if windows_size == 2:
                    start = j if j < len(data[i])-1 else (j-1)
                    end = (j+2) if j < len(data[i])-1 else (j+1)

                elif windows_size == 4:
                    diff = len(data[i]) - j
                    start = j if j < len(data[i])-4 else (j-(4-diff))
                    end = (j+4) if j < len(data[i])-4 else (j+diff)

                elif windows_size == 6:
                    diff = len(data[i]) - j
                    start = j if j < len(data[i])-6 else (j-(6-diff))
                    end = (j+6) if j < len(data[i])-6 else (j+diff)

                else:
                    return None

                for k in range(start, end, 1):
                    windows_data = np.append(windows_data, data[i][k])

                # print(windows_data, windows_data.shape)
                array = DataCombine.extract_feature(windows_data).reshape(-1, 8)
                if len(feature) == 0:
                    feature = array

                else:
                    feature = np.concatenate((feature, array), axis=0)
            # print('feature', feature, feature.shape)
            if len(result) == 0:
                result = feature
            else:
                result = np.concatenate((result, feature), axis=1)
        # print('result', result, result.shape)
        return result

    @staticmethod
    def extract_feature(data):
        feature = np.array([])
        feature = np.append(feature, Formula.means(data))
        feature = np.append(feature, Formula.energy(data))
        feature = np.append(feature, Formula.rms(data))
        feature = np.append(feature, Formula.variance(data))
        feature = np.append(feature, Formula.mad(data))
        feature = np.append(feature, Formula.standard_deviation(data))
        feature = np.append(feature, Formula.maximum(data))
        feature = np.append(feature, Formula.minimum(data))
        return feature


    # def __cal_2_datasets(self):
    #     for i in range(0, len(self.column_data), 1):
    #         dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11 = \
    #                 [], [], [], [], [], [], [], [], [], [], []
    #         # moving windows size is 2
    #         if i < len(self.column_data) - 1:
    #             start = i
    #             end = i + 2
    #         else:
    #             start = i - 1
    #             end = i + 1
    #         for j in range(start, end, 1):
    #             dim1.append(self.column_data[j].acc_x)
    #             dim2.append(self.column_data[j].acc_y)
    #             dim3.append(self.column_data[j].acc_z)
    #             dim4.append(self.column_data[j].gyro_x)
    #             dim5.append(self.column_data[j].gyro_y)
    #             dim6.append(self.column_data[j].gyro_z)
    #             dim7.append(self.column_data[j].mag_x)
    #             dim8.append(self.column_data[j].mag_y)
    #             dim9.append(self.column_data[j].mag_z)
    #             dim10.append(self.column_data[j].pre_x)
    #             dim11.append(self.column_data[j].pre_y)
    #
    #         data = self.__get_characteristic(dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11)
    #         self.dataSet_2.append(data)
    #
    # def __cal_4_datasets(self):
    #     for i in range(0, len(self.column_data), 2):
    #         dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11 = \
    #                 [], [], [], [], [], [], [], [], [], [], []
    #         # moving windows size is 4
    #         if i < len(self.column_data) - 4:
    #             start = i
    #             end = i + 4
    #         else:
    #             diff = len(self.column_data) - i
    #             start = i - (4 - diff)
    #             end = i + diff
    #         for j in range(start, end, 1):
    #             dim1.append(self.column_data[j].acc_x)
    #             dim2.append(self.column_data[j].acc_y)
    #             dim3.append(self.column_data[j].acc_z)
    #             dim4.append(self.column_data[j].gyro_x)
    #             dim5.append(self.column_data[j].gyro_y)
    #             dim6.append(self.column_data[j].gyro_z)
    #             dim7.append(self.column_data[j].mag_x)
    #             dim8.append(self.column_data[j].mag_y)
    #             dim9.append(self.column_data[j].mag_z)
    #             dim10.append(self.column_data[j].pre_x)
    #             dim11.append(self.column_data[j].pre_y)
    #
    #         data = self.__get_characteristic(dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11)
    #         for _ in range(2):
    #             self.dataSet_4.append(data)

    # def __cal_6_datasets(self):
    #     for i in range(0, len(self.column_data), 3):
    #         dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11 = \
    #                 [], [], [], [], [], [], [], [], [], [], []
    #         # moving windows size is 6
    #         if i < len(self.column_data) - 6:
    #             start = i
    #             end = i + 6
    #         else:
    #             diff = len(self.column_data) - i
    #             start = i - (6 - diff)
    #             end = i + diff
    #         for j in range(start, end, 1):
    #             dim1.append(self.column_data[j].acc_x)
    #             dim2.append(self.column_data[j].acc_y)
    #             dim3.append(self.column_data[j].acc_z)
    #             dim4.append(self.column_data[j].gyro_x)
    #             dim5.append(self.column_data[j].gyro_y)
    #             dim6.append(self.column_data[j].gyro_z)
    #             dim7.append(self.column_data[j].mag_x)
    #             dim8.append(self.column_data[j].mag_y)
    #             dim9.append(self.column_data[j].mag_z)
    #             dim10.append(self.column_data[j].pre_x)
    #             dim11.append(self.column_data[j].pre_y)
    #
    #         data = self.__get_characteristic(dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11)
    #         for _ in range(3):
    #             self.dataSet_6.append(data)

    # def __get_characteristic(self, *argv):
    #     all_data, data = [], []
    #     for arg in argv:
    #         # Data.clear()
    #         # print(arg)
    #         data.append(Formula.cal_means(arg))
    #         data.append(Formula.cal_energy(arg))
    #         data.append(Formula.cal_rms(arg))
    #         data.append(Formula.cal_variance(arg))
    #         data.append(Formula.cal_abd(arg))
    #         data.append(Formula.cal_standard_deviation(arg))
    #         data.append(Formula.cal_maximum(arg))
    #         data.append(Formula.cal_minmum(arg))
    #         # all_data.append(Data)
    #     return data

    # def __normalization(self, data):
    #     dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11, dim12 = \
    #         [], [], [], [], [], [], [], [], [], [], [], []
    #     for i in range(0, len(data), 1):
    #         dim1.append(data[i].acc_x)
    #         dim2.append(data[i].acc_y)
    #         dim3.append(data[i].acc_z)
    #         dim4.append(data[i].gyro_x)
    #         dim5.append(data[i].gyro_y)
    #         dim6.append(data[i].gyro_z)
    #         dim7.append(data[i].mag_x)
    #         dim8.append(data[i].mag_y)
    #         dim9.append(data[i].mag_z)
    #         dim10.append(data[i].pre_x)
    #         dim11.append(data[i].pre_y)
    #         dim12.append(data[i].label)
    #     n1 = self.__norm(dim1)
    #     n2 = self.__norm(dim2)
    #     n3 = self.__norm(dim3)
    #     n4 = self.__norm(dim4)
    #     n5 = self.__norm(dim5)
    #     n6 = self.__norm(dim6)
    #     n7 = self.__norm(dim7)
    #     n8 = self.__norm(dim8)
    #     n9 = self.__norm(dim9)
    #     n10 = self.__norm(dim10)
    #     n11 = self.__norm(dim11)
    #     tmp = []
    #     for i in range(len(n1)):
    #         tmp.append(Data(0, 0, n1[i], n2[i], n3[i],
    #                         n4[i], n5[i], n6[i], n7[i],
    #                         n8[i], n9[i], n10[i], n11[i], 0, dim12[i]))
    #     return tmp

    # @staticmethod
    # def __norm(interval):
    #     data = []
    #     for e in interval:
    #         data.append(
    #             (e - min(interval)) / (max(interval) - min(interval)))
    #     return data

    # @staticmethod
    # def __pre_processing(data):
    #     sensor_dim = []
    #     # print(data.shape)
    #     for i in range(len(data)):
    #         sensor_dim.append(
    #             Data(0, 0, data[i][0], data[i][1], data[i][2],
    #                  data[i][3], data[i][4], data[i][5],
    #                  data[i][6], data[i][7], data[i][8],
    #                  data[i][9], data[i][10], 0, 0))
    #     return sensor_dim
