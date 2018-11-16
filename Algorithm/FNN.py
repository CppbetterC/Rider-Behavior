import math
import numpy as np
import copy


class FNN:

    def __init__(self, input_size=0, membership_size=0, rule_size=0, output_size=0,
                 mean=np.array([]), stddev=np.array([]), weight=np.array([]), lr=0.0, l_type=0):
        if input_size != 0:
            # print('Create a Fuzzy Neural Network with type = ', l_type)
            self.input_size = input_size
            self.membership_size = membership_size
            self.rule_size = rule_size
            self.output_size = output_size
            self.category = 6
            self.label_type = l_type

            #####################################################
            # Random Generate the mean, standard deviation
            # mean, stddev, weight
            # Need to be store to describe this neural network
            #####################################################
            self.mean = mean
            self.stddev = stddev
            self.weight = weight

            self.lr = lr
            self.inputs = np.array([])
            self.membership2rule = np.array([])
            self.rule = np.array([])

            self.__error_list = np.array([])

    def forward(self, inputs):
        self.inputs = np.array(inputs)

        # print('input_layer', inputs)

        self.membership2rule = np.array([])
        self.rule = np.zeros(self.rule_size)
        for i in range(len(inputs)):
            gaussian = self.gaussian_func(inputs[i], i)
            self.membership2rule = np.append(self.membership2rule, np.array(gaussian))

        # print('membership_layer', self.membership2rule)

        # Original
        # for i in range(self.rule_size):
        #     self.rule[i] = self.membership2rule[i % 18] *\
        #                    self.membership2rule[(i + 6) % 18] *\
        #                    self.membership2rule[(i + 12) % 18]

        # Modified
        for i in range(self.rule_size):
            values = 1.0
            for j in range(int(self.membership_size / self.input_size)):
                values = values * self.membership2rule[i + j * self.category]
            self.rule[i] = values


        # print('rule_layer', self.rule)

        # rule layer matrices multiple weight
        output = np.matmul(
            self.rule, self.weight.reshape(self.rule_size, -1))

        # print('output_layer', output[0])

        return output[0]

    def backward(self, nn_output, current_label):
        # If the label of the NN output is the current label
        # This denote the output value is closed to the number 1,
        # The other is closed the number -1
        # We will calculate the error value

        if current_label == self.label_type:
            error = 1.0 - nn_output

        else:
            error = (-1.0) - nn_output

        # Record the diff, Observe the diff plot graph
        diff = (error ** 2) / 2
        # print('label:', current_output, 'diff', diff)
        self.__error_list = np.append(self.__error_list, diff)

        copy_mean = copy.deepcopy(self.mean)
        copy_stddev = copy.deepcopy(self.stddev)
        copy_rule = copy.deepcopy(self.rule)
        copy_weight = copy.deepcopy(self.weight)

        # print('copy_mean', copy_mean)
        # print('copy_stddev', copy_stddev)
        # print('copy_weight', copy_weight)

        # Back propagation weight
        for i in range(len(self.weight)):
            self.weight[i] = copy_weight[i] + (self.lr * error * copy_rule[i])
        # print('self.weight', self.weight)

        # Back propagation mean
        # Use for loop to update the value

        # print('copy_mean', copy_mean)
        product = np.array([])
        count = 0
        limit = self.membership_size / self.input_size
        # limit -> 6
        for i in range(len(copy_mean)):
            # if i < 6:
            #     idx = 0
            # elif i > 11:
            #     idx = 2
            # else:
            #     idx = 1

            if i != 0 and (i % limit) == 0:
                count += 1
            idx = count

            tmp = FNN.mean_partial_differential(self.inputs[idx], copy_mean[i], copy_stddev[i])
            product = np.append(product, tmp)

        # print('member', self.membership2rule)
        # print('product', product)
        # print('means', copy_mean)

        for i in range(len(self.mean)):
            values = 1.0
            for j in range(int(self.membership_size / self.input_size)):
                values = values * self.membership2rule[(i + j * 6)]
            self.mean[i] = \
                copy_mean[i] + (self.lr * error * self.weight[i % self.rule_size] * product[i] * values)

            # self.mean[i] = copy_mean[i] + (
            #         self.lr * error * self.weight[i % 6] *
            #         self.membership2rule[(i + 6) % 18] *
            #         self.membership2rule[(i + 12) % 18] * product[i] * self.membership2rule[i % 18])

        product = np.array([])
        count = 0
        limit = self.membership_size / self.input_size
        for i in range(len(copy_stddev)):
            # if i < 6:
            #     idx = 0
            # elif i > 11:
            #     idx = 2
            # else:
            #     idx = 1

            if i != 0 and (i % limit) == 0:
                count += 1
            idx = count

            tmp = FNN.stddev_partial_differential(self.inputs[idx], copy_mean[i], copy_stddev[i])
            product = np.append(product, tmp)

        for i in range(len(self.stddev)):
            values = 1.0
            for j in range(int(self.membership_size / self.input_size)):
                values = values * self.membership2rule[(i + j * 6)]
            self.stddev[i] = \
                copy_stddev[i] + (self.lr * error * self.weight[i % self.rule_size] * product[i] * values)

            # self.stddev[i] = copy_stddev[i] + (
            #         self.lr * error * self.weight[i % 6] *
            #         self.membership2rule[(i + 6) % 18] *
            #         self.membership2rule[(i + 12) % 18] * product[i] * self.membership2rule[i % 18])

            if self.stddev[i] <= 0.1:
                self.stddev[i] = 0.1

        # print('self.stddev', self.stddev)
        # print('--------------------------------------')

    def gaussian_func(self, data, idx):
        tmp = []
        # range(6) represent that every input dimension Labeling
        # will transform to six nn label
        begin = idx * self.category
        for i in range(begin, begin + 6, 1):
            molecular = (-1) * ((data - self.mean[i]) ** 2)
            denominator = self.stddev[i] ** 2
            tmp_value = molecular / denominator
            gaussian_conversion = math.exp(tmp_value)
            tmp.append(gaussian_conversion)
        return tmp

    def training_model(self, epoch, train_data, train_label):
        for i in range(epoch):
            # print('epoch:', i)
            for data, label in zip(train_data, train_label):
                output = self.forward(data)
                self.backward(output, label)

    def testing_model(self, data):
        output_list = []
        for e in data:
            output = self.forward(e)
            output_list.append(output)
        return np.array(output_list)

    @staticmethod
    def mean_partial_differential(x, mean, stddev):
        return (2 * (x - mean)) / (stddev ** 2)

    @staticmethod
    def stddev_partial_differential(x, mean, stddev):
        return (2 * ((x - mean) ** 2)) / (stddev ** 3)

    @property
    def error_list(self):
        return self.__error_list
