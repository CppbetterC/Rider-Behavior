import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class OutputScatter3D:

    @staticmethod
    def output_scatter_3d(input_data, output_data, threshold):
        # plt scatter-3D
        fig = plt.figure()
        ax1 = Axes3D(fig)
        x_data, y_data, z_data = (np.array([]) for _ in range(3))
        for data, output in zip(input_data, output_data):
            # print('data', data)
            # print('output', output)
            if output >= threshold:
                x_data = np.append(x_data, data[0])
                y_data = np.append(y_data, data[1])
                z_data = np.append(z_data, data[2])

        ax1.scatter(x_data, y_data, z_data, s=20)

        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_title('NN Scatter3D')
        plt.show()
