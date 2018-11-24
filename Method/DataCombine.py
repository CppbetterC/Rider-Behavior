import numpy as np

from sklearn import preprocessing

from Method.Formula import Formula
from Method.Normalize import Normalize


class DataCombine:

    @staticmethod
    def combine(org_data):
        print('<---Behavior Distinguish--->')
        # print(org_data)

        # min_max_scalar = preprocessing.MinMaxScaler()
        # normalized_data = min_max_scalar.fit_transform(org_data.astype('float64'))

        normalized_data = Normalize.normalization(org_data).astype('float64')
        normalized_data = normalized_data.T
        # print(normalized_data, normalized_data.shape)

        # windows2, windows4, windows6 = (np.array([]) for _ in range(3))
        windows2 = DataCombine.__moving_windows(normalized_data, 2, 1)
        windows4 = DataCombine.__moving_windows(normalized_data, 4, 2)
        windows6 = DataCombine.__moving_windows(normalized_data, 6, 3)

        result = np.concatenate((windows2, windows4, windows6), axis=1)
        # print('result', result, result.shape)
        return result

    @staticmethod
    def __moving_windows(data, windows_size, step):
        result = np.array([])
        for i in range(len(data)):
            feature = np.array([])
            for j in range(0, len(data[i]), step):
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
