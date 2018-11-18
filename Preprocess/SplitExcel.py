import pandas as pd
import numpy as np
import statistics

# 用來整理 excel
# 將多個分散的 excel 整合成一個大的 excel file
# 這裡面的資料還要在用於 dataCombine 使用
# 變成 一個 264 維度的資料集
# 在拿去降維


class SplitExcel:

    def __init__(self, behavior, file_name):
        print('Split behavior = ', behavior, 'on the Excel')
        self.behavior_type = behavior_type
        self.file_name = file_name                  # Store that we want to load
        self.idx = np.array([])                     # Store the label of the sheet on current location
        self.idx_value = np.array([])               # Store the index of the sheet
        self.tmpData = np.array([])                 # Store the single sheet Labeling
        self.sp_data = np.array([])                 # Store all split Labeling to the excel
        self.data_column = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ',
                            'MagX', 'MagY', 'MagZ',
                            'PreX', 'PreY', 'Label']
        self.export_data = pd.DataFrame(columns=self.data_column)
        self.__load_analysis_excel()

        # Output the data
        # Don't pass to the class Export
        self.export_data.to_excel('../Data/Labeling/' + 'C' + '/' + 'Split_data.xlsx', sheet_name='Data', index=False)

    def __load_analysis_excel(self):
        for fn in self.file_name:
            print('The file is loaded is ', fn, '.xlsx')
            df = pd.read_excel('../Data/LabelingData/' + fn + '.xlsx', sheet_name='分析')
            self.idx_value = df.index
            self.idx = np.array([]).astype(np.int32)
            for i in range(0, len(self.idx_value), 1):
                if self.idx_value[i][0] == 'C':
                    self.idx = np.append(self.idx, int(i))
            s_column = 4
            e_column = 6
            self.tmpData = df.iloc[self.idx[0]: (self.idx[-1] + 1), s_column: e_column]
            self.tmpData = self.tmpData.values

            self.sp_data = np.array([])
            for i in range(len(self.tmpData)):
                tmp = []
                tmp.append(self.idx_value[self.idx[i]])
                tmp.extend(self.tmpData[i])
                self.sp_data = np.append(self.sp_data, np.array(tmp))

            self.sp_data = self.sp_data.reshape(-1, 3)
            # print(self.sp_data)

            df = pd.read_excel('../Data/LabelingData/' + fn + '.xlsx', sheet_name='STSensor')

            for element in self.sp_data:

                # pandas load file will delete the column
                # so, index will be decrease
                tt = df.iloc[int(element[1]) - 2: int(element[2]) - 2 + 1]
                length = (int(element[2]) - 2 + 1) - (int(element[1]) - 2)

                tt.index = range(length)
                tt2 = pd.DataFrame(np.array([element[0] for _ in range(length)]), columns=['Label'])
                tt3 = pd.concat([tt, tt2], axis=1)

                # print(tt3)
                # Remove the unreasonable data by call smooth.py
                # Smooth the self.export_data
                # print('Smooth the unreasonable data')

                org_data = tt3.iloc[:, 2:13].values.T
                org_label = tt3.iloc[:, 14].values.T
                # print(org_data)

                modified_data = np.array([])
                self.header = ['AccX', 'AccY', 'AccZ',
                               'GyroX', 'GyroY', 'GyroZ',
                                'MagX', 'MagY', 'MagZ',
                                'PreX', 'PreY', 'Label']
                # print('org_data\n', org_data)
                for data in org_data:
                    array = SplitExcel.smooth(data)
                    modified_data = np.append(modified_data, array)
                modified_data = np.append(modified_data, org_label)
                modified_data = modified_data.reshape(len(self.header), -1).T
                pd_modified = pd.DataFrame(modified_data, columns=self.header)

                self.export_data = pd.concat([self.export_data, pd_modified], axis=0)

    # Data smoothing
    # mean + one_std_dev * 3
    @staticmethod
    def smooth(data):
        # print(data)
        try:
            one_std_deviation = statistics.stdev(data)
            means = statistics.mean(data)
            up_bound = means + one_std_deviation * 3
            low_bound = means - one_std_deviation * 3
            tmp = []
            for i in range(0, len(data), 1):
                if data[i] > up_bound or data[i] < low_bound:
                    print('smooth')
                    if 0 < i < len(data) - 1:
                        tmp.append((data[i - 1] + data[i + 1]) / 2)
                else:
                    tmp.append(data[i])

        except statistics.StatisticsError:
            print('statistics.StatisticsError', data)
            return data

        return data


file_name = ['074', '091', '094', '108', '119']

# file_name = ['074', '091', '094', '108', '119', '121',
#  '123', '124', '125', '126', '133', '134', '137', '140',
#  '144', '146', '150', '154', '155', '163']
# behavior_type = ['A', 'B', 'C', 'D', 'E']

behavior_type = ['C']
for e in behavior_type:
    sp = SplitExcel(e, file_name)
