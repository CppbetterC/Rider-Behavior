import numpy as np


class Formula:

    @staticmethod
    def means(data):
        return np.mean(data)

    @staticmethod
    def energy(data):
        return np.sum(data**2)/len(data)

    @staticmethod
    # rms -> root mean square
    # 均方根
    def rms(data):
        return (np.sum(data**2)/len(data))**0.5

    @staticmethod
    # variance
    def variance(data):
        mean = np.mean(data)
        return np.sum((data-mean)**2)/(len(data)-1)

    @staticmethod
    # MAD, 平均絕對分差
    def mad(data):
        mean = np.mean(data)
        return np.sum(np.abs(data-mean))/len(data)

    @staticmethod
    def standard_deviation(data):
        return np.std(data)

    @staticmethod
    def maximum(data):
        return np.max(data)

    @staticmethod
    def minimum(data):
        return np.min(data)
