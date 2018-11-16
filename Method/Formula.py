import statistics


class Formula:

    @staticmethod
    def cal_means(data):
        return statistics.mean(data)

    @staticmethod
    def cal_energy(data):
        # print('calculate the energy')
        value = 0.0
        for e in data:
            value += (abs(e)) ** 2
        value = value / len(data)
        return value

    @staticmethod
    # rms -> root mean square
    def cal_rms(data):
        square = []
        for d in data:
            square.append(d*d)
        return (sum(square) / len(data)) ** 0.5

    @staticmethod
    # variance
    def cal_variance(data):
        return statistics.variance(data)

    @staticmethod
    # Average absolute deviation
    def cal_abd(data):
        means = statistics.mean(data)
        value = 0.0
        for e in data:
            value += abs(e - means)
        value = value / len(data)
        return value

    @staticmethod
    def cal_standard_deviation(data):
        return statistics.stdev(data)

    @staticmethod
    def cal_maximum(data):
        return max(data)

    @staticmethod
    def cal_minmum(data):
        return min(data)
