import math
import numpy as np
import copy


class LabelNN:

    def __init__(self, input_size=0, hidden_size=0, output_size=0,
                 weight1=np.array([]), weight2=np.array([]), bias=np.array([]), lr=0.0):

        if input_size != 0:

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size

            self.weight1_size = 36
            self.weight2_size = 36
            self.bias_size = 6

            #####################################################
            # weight1, bias, weight2 (v)
            # Represent the threed different weight
            # need to be store to describe this neural network
            #####################################################

            self.lr = lr
            self.inputs = np.array([])
            self.hidden = np.array([])
            self.output = np.array([])

            self.weight1 = weight1
            self.weight2 = weight2
            self.bias = bias

            self.__error_list = np.array([])
            # self.weight1 = \
            #     np.array([random.uniform(-1, 1) for _ in range(self.weight1_size)]).reshape(-1, 6)
            # self.weight2 = \
            #     np.array([random.uniform(-1, 1) for _ in range(self.weight2_size)]).reshape(-1, 6)
            # self.bias = \
            #     np.array([random.uniform(-1, 1) for _ in range(self.bias_size)])

    def forward(self, inputs):

        self.inputs = np.array(inputs)
        self.hidden_in = np.array([])
        self.hidden_out = np.array([])
        output = np.array([])

        print('self.inputs', self.inputs)
        print('self.weight1', self.weight1)
        print('self.weight2', self.weight2)
        print('self.bias', self.bias)

        try:
            # input to hidden_in
            for i in range(0, self.hidden_size, 1):
                self.hidden_in = np.append(
                    self.hidden_in, (self.inputs.dot(self.weight1[i]) + self.bias[i]))

            print('self.hidden_in', self.hidden_in)

            # hidden_in to hidden_out
            for element in self.hidden_in:
                values = LabelNN.tangent_sigmoid(element)
                self.hidden_out = np.append(self.hidden_out, values)

            print('self.hidden_out', self.hidden_out)

            for i in range(0, self.output_size, 1):
                output = np.append(output, self.hidden_out.dot(self.weight2[i]))

            print('output_layer', output, len(output))

        except OverflowError:
            # print('<----------------------------------------------->')
            # print("<---Values OverFlowError(LabelNN[forward])--->")
            raise

        except TypeError:
            # print('<----------------------------------------------->')
            # print("<---TypeError(LabelNN[forward])--->")
            raise

        return output

    def backward(self, nn_output, label):
        print(label)
        ideal_output = {1: [1, -1, -1, -1, -1, -1], 2: [-1, 1, -1, -1, -1, -1],
                        3: [-1, -1, 1, -1, -1, -1], 4: [-1, -1, -1, 1, -1, -1],
                        5: [-1, -1, -1, -1, 1, -1], 6: [-1, -1, -1, -1, -1, 1]}

        error = np.array(ideal_output[label]) - nn_output

        diff = (error ** 2) / 2
        if not bool(self.__error_list.size):
            self.__error_list = diff.reshape(-1, 6)

        else:
            self.__error_list = np.concatenate([self.__error_list, diff.reshape(-1, 6)])

        # print('label:', label, 'diff', diff)

        copy_weight1 = copy.deepcopy(self.weight1)
        copy_weight2 = copy.deepcopy(self.weight2)
        copy_bias = copy.deepcopy(self.bias)

        # Back propagation weight
        # Modify the weight2
        for i in range(len(self.weight2)):
            for j in range(len(self.weight2[i])):
                self.weight2[i][j] = copy_weight2[i][j] + (self.lr * error[j] * self.hidden_out[i])

        for x, y in zip(copy_weight2, self.weight2):
            print(x, '->', y)

        # print('copy_weight2', )
        # print('self.weight2', )

        # Modify the bias
        for i in range(self.bias_size):
            # accumulation the error(1) ~ error(6)
            accumulation = 0.0
            for j in range(len(error)):
                try:
                    accumulation += error[j] * (-1) * copy_weight2[i][j] * \
                                    LabelNN.sigmoid_differential(self.hidden_in[i])
                except ZeroDivisionError:
                    # print('<----------------------------------------------->')
                    # print("<---ZeroDivisionError(LabelNN[backward])--->")
                    raise

            self.bias[i] = copy_bias[i] + (-1 * self.lr * accumulation)

        for x, y in zip(copy_bias, self.bias):
            print(x, '->', y)

        # print('copy_bias', copy_bias)
        # print('self.bias', self.bias)

        # Modify the weight1
        for i in range(len(self.weight1)):
            for j in range(len(self.weight1[i])):
                accumulation = 0.0
                for k in range(len(error)):
                    accumulation += error[k] * (-1) * copy_weight2[i][k] *\
                                    LabelNN.sigmoid_differential(self.hidden_in[i]) * self.inputs[j]
                self.weight1[i][j] = copy_weight1[i][j] + (-1 * self.lr * accumulation)

        for x, y in zip(copy_weight1, self.weight1):
            print(x, '->', y)

    def training_model(self, epoch, train_data, train_label):
        try:
            for i in range(epoch):
                # print('epoch:', i)
                output = self.forward(train_data)
                self.backward(output, train_label)
        except OverflowError:
            # print('<----------------------------------------------->')
            # print("<---Values OverFlowError(LabelNN[training_model])--->")
            raise
        except ZeroDivisionError:
            # print('<----------------------------------------------->')
            # print("<---ZeroDivisionError(LabelNN[training_model])--->")
            raise

    def testing_model(self, data):
        try:
            output_list = np.array([])
            for e in data:
                output = self.forward(e)
                output_list = np.append(output_list, output)

        except OverflowError:
            # print('<----------------------------------------------->')
            # print("<---Values OverFlowError(LabelNN[testing_model])--->")
            raise

        return output_list.reshape(-1, 6)

    # Activation function
    @staticmethod
    def tangent_sigmoid(data):
        try:
            values = (math.exp(data) - math.exp(-data)) / (math.exp(data) + math.exp(-data))
        except OverflowError:
            # print('<----------------------------------------------->')
            # print("<---Values OverFlowError(LabelNN[tangent_sigmoid])--->")
            raise
        return values

    @staticmethod
    def sigmoid_differential(data):
        try:
            values = 4 / ((math.exp(data) + math.exp(data)) ** 2)
        except OverflowError:
            # print('<----------------------------------------------->')
            # print("<---Values OverFlowError(LabelNN[sigmoid_differential])--->")
            raise
        except ZeroDivisionError:
            # print('<----------------------------------------------->')
            # print("<---ZeroDivisionError(LabelNN[sigmoid_differential])--->")
            raise
        return values

    # Pick the max value and assign the corresponding index
    @staticmethod
    def label_encode(data):
        result = np.array([])
        for array in data:
            idx = np.argmax(array)
            result = np.append(result, (idx + 1))
        return result

    @property
    def error_list(self):
        return self.__error_list
