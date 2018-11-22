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