import os
import numpy as np
import matplotlib.pyplot as plt


class ErrorPlot:

    """
    para1 -> title
    para2 -> x axis data (random nn)
    para3 -> y axis data (nn accuracy)
    """

    @staticmethod
    def error_trend(title, x_data, y_data, path=''):
        plt.figure(figsize=(8, 6), dpi=100)
        plt.title(title)
        plt.xlabel('Modified Times')
        plt.ylabel('Diff')
        plt.xlim(0, x_data)
        plt.ylim(0, 1)
        plt.plot(np.array([i for i in range(x_data)]), y_data, marker='o')
        plt.savefig(path)
        plt.ion()
        plt.pause(5)
        plt.close()

    @staticmethod
    def mul_error_trend(title, x_data, y_data, path=''):
        color_list = ['r', 'b', 'g', 'y', 'c', 'silver']
        x_new = np.array([i for i in range(x_data)])
        y_new = y_data.T

        for i in range(len(color_list)):

            # Picture Element
            plt.figure(figsize=(8, 6), dpi=100)
            plt.title(title)
            plt.xlabel('Modified Times')
            plt.ylabel('Diff')
            plt.xlim(0, x_data)
            plt.ylim(0, 1)

            # Link path
            rel_path = path + '\\C' + str(i+1) + '_Best_LNN_error_trend.png'
            abs_path = os.path.join(os.path.dirname(__file__), rel_path)

            plt.plot(x_new, y_new[i], marker='o', color=color_list[i])
            # plt.plot(x_new, y_new[1], marker='o', color=color_list[1])
            # plt.plot(x_new, y_new[2], marker='o', color=color_list[2])
            # plt.plot(x_new, y_new[3], marker='o', color=color_list[3])
            # plt.plot(x_new, y_new[4], marker='o', color=color_list[4])
            # plt.plot(x_new, y_new[5], marker='o', color=color_list[5])
            plt.savefig(abs_path )
            plt.ion()
            plt.pause(5)
            plt.close()
