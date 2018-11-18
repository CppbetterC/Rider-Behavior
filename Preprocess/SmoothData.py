# import statistics
# import numpy as np
# import pandas as pd
#
# from Method.Export import Export
# from Method.LoadData import LoadData
# from Method.SensorData import Data
#
#
# def pre_processing(data):
#     length = len(data[0])
#     tmp, tmp2, tmp3 = [], [], []
#     if length == 0:
#         print('There don\'t have this Data format with length=0')
#         return
#     if length == 2:
#         for e in data:
#             tmp.append(float(e[0]))
#             tmp2.append(float(e[1]))
#         return tmp, tmp2
#     elif length == 3:
#         for e in data:
#             tmp.append(float(e[0]))
#             tmp2.append(float(e[1]))
#             tmp3.append(float(e[2]))
#         return tmp, tmp2, tmp3
#     else:
#         print('There don\'t have this Data format')
#         return
#
# # Data smoothing
# # mean + one_std_dev * 3
# def smoothing(data):
#     one_std_deviation = statistics.stdev(data)
#     means = statistics.mean(data)
#     up_bound = means + one_std_deviation * 3
#     low_bound = means - one_std_deviation * 3
#     tmp = []
#     for i in range(0, len(data), 1):
#         if data[i] > up_bound or data[i] < low_bound:
#             print('smooth')
#             if 0 < i < len(data) - 1:
#                 tmp.append((data[i-1] + data[i+1]) / 2)
#         else:
#             tmp.append(data[i])
#     return data
#
#
# # load the original Data
# def split_org_data(file):
#     load = LoadData()
#     org_data = load.get_stsensor_excel(file)
#     acc, gyro, mag, pre, hall = [], [], [], [], []
#     # acc_x, acc_y, acc_z = [], [], []
#     # gyro_x, gyro_y, gyro_z = [], [], []
#     # mag_x, mag_y, mag_z = [], [], []
#     # pre_x, pre_y = [], []
#     for e in org_data:
#         # print(line.Date, line.time, line.Acc, line.Gyro, line.Mag, line.Pre, line.Hall)
#         acc.append(e.Acc)
#         gyro.append(e.Gyro)
#         mag.append(e.Mag)
#         pre.append(e.Pre)
#         hall.append(e.Hall)
#
#     # Data pre_processing
#     acc_x, acc_y, acc_z = pre_processing(acc)
#     gyro_x, gyro_y, gyro_z = pre_processing(gyro)
#     mag_x, mag_y, mag_z = pre_processing(mag)
#     pre_x, pre_y = pre_processing(pre)
#
#     # Data smoothing
#     acc_x_ = smoothing(acc_x)
#     acc_y_ = smoothing(acc_y)
#     acc_z_ = smoothing(acc_z)
#
#     gyro_x_ = smoothing(gyro_x)
#     gyro_y_ = smoothing(gyro_y)
#     gyro_z_ = smoothing(gyro_z)
#
#     mag_x_ = smoothing(mag_x)
#     mag_y_ = smoothing(mag_y)
#     mag_z_ = smoothing(mag_z)
#
#     pre_x_ = smoothing(pre_x)
#     pre_y_ = smoothing(pre_y)
#
#     hall_ = [float(x) for x in hall]
#     sensor_data = np.array([])
#     # for i in range(0, len(org_data), 1):
#     #     sensor_data.append(Data(org_data[i].date, org_data[i].time, acc_x_[i], acc_y_[i], acc_z_[i],
#     #                             gyro_x_[i], gyro_y_[i], gyro_z_[i], mag_x_[i], mag_y_[i], mag_z_[i],
#     #                             pre_x_[i], pre_y_[i], hall_[i]))
#
#     t_date = np.array([])
#     t_time = np.array([])
#     for i in range(0, len(org_data), 1):
#         t_date = np.append(t_date, org_data[i].date)
#         t_time = np.append(t_time, org_data[i].time)
#
#     sensor_data = np.append(sensor_data, t_date)
#     sensor_data = np.append(sensor_data, t_time)
#     sensor_data = np.append(sensor_data, acc_x_)
#     sensor_data = np.append(sensor_data, acc_y_)
#     sensor_data = np.append(sensor_data, acc_z_)
#     sensor_data = np.append(sensor_data, gyro_x_)
#     sensor_data = np.append(sensor_data, gyro_y_)
#     sensor_data = np.append(sensor_data, gyro_z_)
#     sensor_data = np.append(sensor_data, mag_x_)
#     sensor_data = np.append(sensor_data, mag_y_)
#     sensor_data = np.append(sensor_data, mag_z_)
#     sensor_data = np.append(sensor_data, pre_x_)
#     sensor_data = np.append(sensor_data, pre_y_)
#     sensor_data = np.append(sensor_data, hall_)
#     sensor_data = sensor_data.reshape(-1, len(t_date)).T
#     return sensor_data
#
#
# file_list = ['074', '091', '094', '108', '119']
# header = ['Date', 'time', 'AccX', 'AccY', 'AccZ',
#           'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ',
#           'PreX', 'PreY', 'Hall']
# for file in file_list:
#     data = split_org_data(file)
#     print(data)
#     pd_data = pd.DataFrame(data, columns=header)
#     print(pd_data.head())
#     Export.export_split_data(data, file)
#
#     print('output end')
