import os
import csv
import numpy as np
import pandas as pd

from openpyxl import Workbook
from openpyxl import load_workbook


class Export:

    @staticmethod
    def export_split_data(data, file):
        path_name = '../Data/LabelingData(Smooth)/' + file + '.xlsx'
        path_name = os.path.join(os.path.dirname(__file__), path_name)
        header = ['Date', 'time', 'AccX', 'AccY', 'AccZ',
                  'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ',
                  'PreX', 'PreY', 'Hall']
        df = pd.DataFrame(data, columns=header)
        writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='STSensor', index=False)
        writer.save()


    @staticmethod
    def export_dimension_data(data, type=0):
        if type == 2:
            path_name = '../Data/DimensionDataSets/2_set/d_074.txt'
        elif type == 4:
            path_name = '../Data/DimensionDataSets/4_set/d_074.txt'
        elif type == 6:
            path_name = '../Data/DimensionDataSets/6_set/d_074.txt'
        else:
            print('Error')
            path_name = ''
        path_name = os.path.join(os.path.dirname(__file__), path_name)
        with open(path_name, "a") as f:
            for e in data:
                f.write(str(e) + ';')
            f.write('\n')
        f.close()

    @staticmethod
    def export_comdimension_data(data, file_name):
        dim_data = []
        windows = ['2', '4', '6']
        excel_row = ['acc_x', 'acc_y', 'acc_z',
                     'gyro_x', 'gyro_y', 'gyro_z',
                     'mag_x', 'mag_y', 'mag_z', 'pre_x', 'pre_y_']
        feature = ['means', 'energy', 'rms', 'variance', 'mad', 'std_deviation', 'max', 'min']
        # i -> windows size(2, 4, 6)
        # j -> dimension in the excel
        # ele -> calculated feature by formula
        for i in windows:
            for j in excel_row:
                for ele in feature:
                    dim_data.append(i + '-' + j + '-' + ele)
        path_name = '../Data/Combine_STsensor_data/' + file_name + '.csv'
        path_name = os.path.join(os.path.dirname(__file__), path_name)
        with open(path_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(dim_data)
            for tmp in data:
                writer.writerow([float(e) for e in tmp])
        csvfile.close()

    @staticmethod
    def export_knn(traindata, realdata, file_name):
        path_name = '../Data/Combine_STsensor_data/' + file_name + 'knn' + '.xlsx'
        path_name = os.path.join(os.path.dirname(__file__), path_name)
        ws = Workbook()
        wb = ws.active
        wb.append(['train', 'real'])
        for i in range(len(traindata)):
            wb.append([traindata[i], realdata[i]])
        ws.save(path_name)

    @staticmethod
    def export_2dimension_data(data, labels, filename):
        path_name = '../Data/2dimension(' + filename + ').txt'
        path_name = os.path.join(os.path.dirname(__file__), path_name)
        tmp = data.tolist()
        with open(path_name, 'w') as f:
            for i in range(len(tmp)):
                for e in tmp[i]:
                    f.write(str(e) + ';')
                f.write(str(labels.tolist()[i]))
                f.write('\n')
        f.close()

    # @staticmethod
    # def export_refactor_data_txt(data, label):
    #     path_name = '../Data/Labeling/C/Refactor_SpData_C' + str(label) + '.txt'
    #     path_name = os.path.join(os.path.dirname(__file__), path_name)
    #     with open(path_name, 'a', encoding='utf-8') as f:
    #         for i in range(len(data)):
    #             for j in range(len(data[i])):
    #                 if i < len(data[i]) - 1:
    #                     f.write(str(data[i][j]))
    #                     f.write(' ')
    #                 else:
    #                     f.write(str(data[i][j]))
    #             f.write('\n')
    #     f.close()

    @staticmethod
    def export_refactor_data_excel(data, file_num, label):
        path_name = '../Data/Labeling/C/Refactor_SpData_C' + str(file_num) + '.xlsx'
        path_name = os.path.join(os.path.dirname(__file__), path_name)
        excel_data = {}
        x_value, y_value, z_value = ([] for _ in range(3))
        for i in range(len(data)):
            x_value.append(data[i][0])
            y_value.append(data[i][1])
            z_value.append(data[i][2])
        excel_data['Dim1'] = x_value
        excel_data['Dim2'] = y_value
        excel_data['Dim3'] = z_value
        excel_data['Label'] = [label for _ in range(len(data))]

        df = pd.DataFrame(excel_data)

        try:
            tmp_data = pd.read_excel(path_name)
            writer = pd.ExcelWriter(path_name)
            result = pd.concat([tmp_data, df], ignore_index=True)
            result.to_excel(writer, sheet_name='Labeling_Data', index=False)

        except FileNotFoundError:
            writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Labeling_Data', index=False)

        writer.save()

    @staticmethod
    def export_fnn_output(data, file_num):
        path = 'C:\\Users\\yuren\\Desktop\\20180923輸出\\FNN_Output_C' + str(file_num) + '.xlsx'
        excel_data = {}
        excel_data['C1'] = data[0]
        excel_data['C2'] = data[1]
        excel_data['C3'] = data[2]
        excel_data['C4'] = data[3]
        excel_data['C5'] = data[4]
        excel_data['C6'] = data[5]

        df =pd.DataFrame(excel_data)
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        df.to_excel(writer, index=False)

        writer.save()

    # # This method export those data
    # # was decrease the dimension by LDA algorithm
    # @staticmethod
    # def export_lda_data(data, path_name):
    #     column = ['Dim1', 'Dim2', 'Dim3', 'label']
    #     df = pd.DataFrame(data, columns=column)
    #     writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
    #     df.to_excel(writer, sheet_name='Labeling_Data', index=False)
    #     writer.save()

    # This method export those data
    # Those were expand by SVM-SMOTE algorithm
    @staticmethod
    def export_balanced_data(data, path_name):
        column = ['Dim1', 'Dim2', 'Dim3', 'label']
        df = pd.DataFrame(data, columns=column)
        writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Labeling_Data', index=False)
        writer.save()

    @staticmethod
    def export_lle_data(data, path_name):
        column = ['Dim1', 'Dim2', 'Dim3', 'label']
        df = pd.DataFrame(data, columns=column)
        writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Labeling_Data', index=False)
        writer.save()

    @staticmethod
    def export_org_data(data, path_name):
        header = ['Dim' + str(i) for i in range(1, 265, 1)]
        header.append('Label')
        df = pd.DataFrame(data, columns=header)
        writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Labeling_Data', index=False)
        writer.save()

    @staticmethod
    def export_fnn_train_data(data, path_name):
        header = ['Dim' + str(i) for i in range(1, 265, 1)]
        header.append('Label')
        df = pd.DataFrame(data, columns=header)
        writer = pd.ExcelWriter(path_name, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Labeling_Data', index=False)
        writer.save()



