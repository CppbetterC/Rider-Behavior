import numpy as np


class Normalize:

    @staticmethod
    def normalization(data):
        new_data = np.array([])
        tmp = data.T
        length = len(tmp[0])
        for array in tmp:
            sub_array = []
            max_value = max(array)
            min_value = min(array)
            for element in array:
                sub_array.append(2 * ((element - min_value) / (max_value - min_value)) - 1)
            new_data = np.append(new_data, sub_array)
        new_data = new_data.reshape(-1, length).T
        return new_data


"""
正規化 mehtod 1

# normalized_data = preprocessing.normalize(org_data, norm='l1')
# print(normalized_data)
min_max_scaler = preprocessing.MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(org_data)

# reduced_data = ra.nca(normalized_data, org_label, dim)
# reduced_data = ra.lfda(normalized_data, org_label, dim)
reduced_data = ra.mlkr(normalized_data, org_label, dim)

# 呼叫不同的降維法去降維, 取特徵直
# normalized_data = preprocessing.normalize(reduced_data, norm='l1')
normalized_data = min_max_scaler. fit_transform(reduced_data)

"""