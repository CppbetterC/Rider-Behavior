from Method.Formula import Formula
from Method.SensorData import Data


class DataCombine:

    def __init__(self, org_data):
        print('<---Behavior Distinguish--->')

        # 將numpy的資料集轉成物件型態
        new_data = self.__pre_processing(org_data)

        # Normalization (正規化)
        self.column_data = self.__normalization(new_data)

        self.dimension_data = []
        self.dataSet_2, self.dataSet_4, self.dataSet_6 = [], [], []
        self.__cal_2_datasets()
        self.__cal_4_datasets()
        self.__cal_6_datasets()

    # combine the dataSet
    def cal_dimension(self):
        data = []
        print('<---Combine the dataSet--->')
        for i in range(0, len(self.dataSet_2), 1):
            tmp = self.dataSet_2[i] + self.dataSet_4[i] + self.dataSet_6[i]
            data.append(tmp)
        return data

    def __cal_2_datasets(self):
        for i in range(0, len(self.column_data), 1):
            dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11 = \
                    [], [], [], [], [], [], [], [], [], [], []
            # moving windows size is 2
            if i < len(self.column_data) - 1:
                start = i
                end = i + 2
            else:
                start = i - 1
                end = i + 1
            for j in range(start, end, 1):
                dim1.append(self.column_data[j].acc_x)
                dim2.append(self.column_data[j].acc_y)
                dim3.append(self.column_data[j].acc_z)
                dim4.append(self.column_data[j].gyro_x)
                dim5.append(self.column_data[j].gyro_y)
                dim6.append(self.column_data[j].gyro_z)
                dim7.append(self.column_data[j].mag_x)
                dim8.append(self.column_data[j].mag_y)
                dim9.append(self.column_data[j].mag_z)
                dim10.append(self.column_data[j].pre_x)
                dim11.append(self.column_data[j].pre_y)

            data = self.__get_characteristic(dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11)
            self.dataSet_2.append(data)

    def __cal_4_datasets(self):
        for i in range(0, len(self.column_data), 2):
            dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11 = \
                    [], [], [], [], [], [], [], [], [], [], []
            # moving windows size is 4
            if i < len(self.column_data) - 4:
                start = i
                end = i + 4
            else:
                diff = len(self.column_data) - i
                start = i - (4 - diff)
                end = i + diff
            for j in range(start, end, 1):
                dim1.append(self.column_data[j].acc_x)
                dim2.append(self.column_data[j].acc_y)
                dim3.append(self.column_data[j].acc_z)
                dim4.append(self.column_data[j].gyro_x)
                dim5.append(self.column_data[j].gyro_y)
                dim6.append(self.column_data[j].gyro_z)
                dim7.append(self.column_data[j].mag_x)
                dim8.append(self.column_data[j].mag_y)
                dim9.append(self.column_data[j].mag_z)
                dim10.append(self.column_data[j].pre_x)
                dim11.append(self.column_data[j].pre_y)

            data = self.__get_characteristic(dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11)
            for _ in range(2):
                self.dataSet_4.append(data)

    def __cal_6_datasets(self):
        for i in range(0, len(self.column_data), 3):
            dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11 = \
                    [], [], [], [], [], [], [], [], [], [], []
            # moving windows size is 6
            if i < len(self.column_data) - 6:
                start = i
                end = i + 6
            else:
                diff = len(self.column_data) - i
                start = i - (6 - diff)
                end = i + diff
            for j in range(start, end, 1):
                dim1.append(self.column_data[j].acc_x)
                dim2.append(self.column_data[j].acc_y)
                dim3.append(self.column_data[j].acc_z)
                dim4.append(self.column_data[j].gyro_x)
                dim5.append(self.column_data[j].gyro_y)
                dim6.append(self.column_data[j].gyro_z)
                dim7.append(self.column_data[j].mag_x)
                dim8.append(self.column_data[j].mag_y)
                dim9.append(self.column_data[j].mag_z)
                dim10.append(self.column_data[j].pre_x)
                dim11.append(self.column_data[j].pre_y)

            data = self.__get_characteristic(dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11)
            for _ in range(3):
                self.dataSet_6.append(data)

    def __get_characteristic(self, *argv):
        all_data, data = [], []
        for arg in argv:
            # Data.clear()
            # print(arg)
            data.append(Formula.cal_means(arg))
            data.append(Formula.cal_energy(arg))
            data.append(Formula.cal_rms(arg))
            data.append(Formula.cal_variance(arg))
            data.append(Formula.cal_abd(arg))
            data.append(Formula.cal_standard_deviation(arg))
            data.append(Formula.cal_maximum(arg))
            data.append(Formula.cal_minmum(arg))
            # all_data.append(Data)
        return data

    def __normalization(self, data):
        dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11, dim12 = \
            [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(0, len(data), 1):
            dim1.append(data[i].acc_x)
            dim2.append(data[i].acc_y)
            dim3.append(data[i].acc_z)
            dim4.append(data[i].gyro_x)
            dim5.append(data[i].gyro_y)
            dim6.append(data[i].gyro_z)
            dim7.append(data[i].mag_x)
            dim8.append(data[i].mag_y)
            dim9.append(data[i].mag_z)
            dim10.append(data[i].pre_x)
            dim11.append(data[i].pre_y)
            dim12.append(data[i].label)
        n1 = self.__norm(dim1)
        n2 = self.__norm(dim2)
        n3 = self.__norm(dim3)
        n4 = self.__norm(dim4)
        n5 = self.__norm(dim5)
        n6 = self.__norm(dim6)
        n7 = self.__norm(dim7)
        n8 = self.__norm(dim8)
        n9 = self.__norm(dim9)
        n10 = self.__norm(dim10)
        n11 = self.__norm(dim11)
        tmp = []
        for i in range(len(n1)):
            tmp.append(Data(0, 0, n1[i], n2[i], n3[i],
                            n4[i], n5[i], n6[i], n7[i],
                            n8[i], n9[i], n10[i], n11[i], 0, dim12[i]))
        return tmp

    @staticmethod
    def __norm(interval):
        data = []
        for e in interval:
            data.append(
                (e - min(interval)) / (max(interval) - min(interval)))
        return data

    @staticmethod
    def __pre_processing(data):
        sensor_dim = []
        # print(data.shape)
        for i in range(len(data)):
            sensor_dim.append(
                Data(0, 0, data[i][0], data[i][1], data[i][2],
                     data[i][3], data[i][4], data[i][5],
                     data[i][6], data[i][7], data[i][8],
                     data[i][9], data[i][10], 0, 0))
        return sensor_dim
