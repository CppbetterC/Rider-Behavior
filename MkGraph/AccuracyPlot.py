import numpy as np
import matplotlib.pyplot as plt


class AccuracyPlot:

    """
    para1 -> title
    para2 -> x axis data (random nn)
    para3 -> y axis data (nn accuracy)
    """
    @staticmethod
    def build_accuracy_plot(title, x_data, y_data, path=''):
        plt.figure(figsize=(8, 6), dpi=80)
        plt.title(title)
        plt.xlabel('Neural Network')
        plt.ylabel('Accuracy')
        plt.xlim(np.min(x_data)-1, np.max(x_data)+1)
        plt.ylim(0, 1)
        plt.plot(x_data, y_data, marker='o')
        y_data = np.array([round(e, 2) for e in y_data])
        for xy in zip(x_data, y_data):
            plt.annotate(
                "%s" % xy[1], xy=xy, xytext=(-5, 5), textcoords='offset points', color='red')
        if not bool(path):
            plt.savefig('./Data/Graph/'+title+'.png')
        else:
            plt.savefig(path)
        plt.ion()
        plt.pause(5)
        plt.close()
