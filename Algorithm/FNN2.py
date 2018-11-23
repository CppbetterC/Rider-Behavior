import numpy as np
import copy

"""
這是另一個模糊類神經的結構，
Input Layer: 5 neuron
Membership Layer: 5*6=30 neuron
Rule Layer: 5 neuron
Output Layer: 6 neuron

實作步驟:
Step 1: 先將 LNN_Train.xlsx 的資料集讀入
Step 2: 用 LDA 降維法去降維度到5維度
Step 3: 將資料集傳入模糊類神經去進行預測
Step 4: 挑選 Output Layer 輸出最高的數值來當作該標籤

mean = [['m0', 'm1', 'm2', 'm3', 'm4', 'm5'],
        ['m6', 'm7', 'm8', 'm9', 'm10', 'm11'],
        ['m12', 'm13', 'm14', 'm15', 'm16', 'm17'],
        ['m18', 'm19', 'm20', 'm21', 'm22', 'm23'],
        ['m24', 'm25', 'm26', 'm27', 'm28', 'm29']])

stddev = [['std0', 'std1', 'std2', 'std3', 'std4', 'std5'],
          ['std6', 'std7', 'std8', 'std9', 'std10', 'std11'],
          ['std12', 'std13', 'std14', 'std15', 'std16', 'std17'],
          ['std18', 'std19', 'std20', 'std21', 'std22', 'std23'],
          ['std24', 'std25', 'std26', 'std27', 'std28', 'std29']]
       
weight = [[ w11,  w12,  w13,  w14,  w15, w16],
          [ w21,  w22,  w23,  w24,  w25, w26],
          [ w31,  w32,  w33,  w34,  w35, w36],
          [ w41,  w42,  w43,  w44,  w45, w46],
          [ w51,  w52,  w53,  w54,  w55, w56],
          [ w61,  w62,  w63,  w64,  w65, w66]])
"""


class FNN2:

    def __init__(self, input_size=0, membership_size=0, rule_size=0, output_size=0, category=0,
                 mean=np.array([]), stddev=np.array([]), weight=np.array([]), lr=0.0, label=0):
        if input_size != 0:
            # print('Create a Fuzzy Neural Network with type = ', l_type)
            self.input_size = input_size
            self.membership_size = membership_size
            self.rule_size = rule_size
            self.output_size = output_size
            self.category = category
            self.train_label = label

            #####################################################
            # Random Generate the mean, standard deviation
            # mean, stddev, weight
            # Need to be store to describe this neural network
            #####################################################
            self.mean = mean
            self.stddev = stddev
            self.weight = weight
            print('Initialization Mean, Stddev, Weight')
            print('mean', self.mean)
            print('stddev', self.stddev)
            print('weight', self.weight)

            self.lr = lr
            self.inputs = np.array([])
            self.membership2rule = np.array([])
            self.rule = np.array([])
            self.__error_list = np.array([])
            self.__loss = 0.0
            self.__loss_list = np.array([])
            self.ideal_output = \
                {1: np.array([1, -1, -1, -1, -1, -1]),
                 2: np.array([-1, 1, -1, -1, -1, -1]),
                 3: np.array([-1, -1, 1, -1, -1, -1]),
                 4: np.array([-1, -1, -1, 1, -1, -1]),
                 5: np.array([-1, -1, -1, -1, 1, -1]),
                 6: np.array([-1, -1, -1, -1, -1, 1])}

    def forward(self, inputs):
        self.inputs = inputs
        self.membership2rule = np.array([])
        self.rule = np.array([])
        # print('input_layer', inputs)

        for i in range(len(self.inputs)):
            gaussian = self.gaussian_func(self.inputs[i], i)
            self.membership2rule = np.append(self.membership2rule, gaussian)
        # print('membership_layer', self.membership2rule)

        membership2rule = self.membership2rule.reshape(-1, self.category).T
        for array in membership2rule:
            self.rule = np.append(self.rule, np.multiply.reduce(array))
        # print('rule_layer', self.rule)

        weight = self.weight.T
        # (1 by 5) dot (5 by 6) -> (1 by 6)
        output = self.rule.dot(weight)
        # output 應該要盡量往1或-1靠攏才是訓練不錯的
        # print('output_layer', output)
        return output

    def backward(self, output, ideal_label):
        error = self.ideal_output[ideal_label] - output

        # Record the diff, Observe the diff plot graph
        diff = np.add.reduce((error ** 2) / 2)
        # print('ideal_label:', ideal_label, 'diff', diff)
        self.__error_list = np.append(self.__error_list, diff)
        # self.__loss += diff

        copy_mean = copy.deepcopy(self.mean)
        copy_stddev = copy.deepcopy(self.stddev)
        copy_rule = copy.deepcopy(self.rule)
        copy_weight = copy.deepcopy(self.weight)

        # print('copy_mean', copy_mean)
        # print('copy_stddev', copy_stddev)
        # print('copy_weight', copy_weight)

        # Back propagation weight
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                self.weight[i][j] = copy_weight[i][j] + (self.lr * error[i] * copy_rule[j])

        # print('self.weight', self.weight)
        # Back propagation mean
        # Use for loop to update the value

        # print('copy_mean', copy_mean)
        """
        Modify Mean
        """
        for i in range(len(copy_mean)):
            for j in range(len(copy_mean[i])):
                # 偏微分計算結果
                differential = FNN2.mean_partial_differential(self.inputs[i], copy_mean[i][j], copy_stddev[i][j])
                # rule layer 連乘的結果
                membership_reduce = copy_rule[(i*self.category+j) % self.category]
                values = 0.0
                for k in range(len(error)):
                    values += (error[k] * (-1) * self.weight[k][i % self.category] * membership_reduce * differential)
                self.mean[i][j] = copy_mean[i][j] + ((-1) * self.lr * values)

        # print('member', self.membership2rule)
        # print('product', product)
        # print('means', copy_mean)

        """
        Modify Stddev
        """
        for i in range(len(copy_stddev)):
            for j in range(len(copy_stddev[i])):
                # 偏微分計算結果
                differential = FNN2.stddev_partial_differential(self.inputs[i], copy_mean[i][j], copy_stddev[i][j])
                # rule layer 連乘的結果
                membership_reduce = copy_rule[(i*self.category+j) % self.category]
                values = 0.0
                for k in range(len(error)):
                    values += (error[k] * (-1) * self.weight[k][i % self.category] * membership_reduce * differential)
                self.stddev[i][j] = copy_stddev[i][j] + ((-1) * self.lr * values)

                if self.stddev[i][j] <= 0.1:
                    self.stddev[i][j] = 0.1

        # print('self.stddev', self.stddev)
        # print('--------------------------------------')

    def gaussian_func(self, data, idx):
        result = np.array([])
        mean = self.mean[idx]
        stddev = self.stddev[idx]
        for x, y in zip(mean, stddev):
            molecular = (-1) * ((data - x) ** 2)
            denominator = y ** 2
            gaussian_conversion = np.exp(molecular/denominator)
            result = np.append(result, gaussian_conversion)
        return result

    def training_model(self, epoch, train_data, train_label):
        for i in range(epoch):
            # print('epoch:', i)
            self.__loss = 0.0
            for data, label in zip(train_data, train_label):
                output = self.forward(data)
                self.backward(output, label)
            self.__loss_list = np.append(self.__loss_list, self.__loss)

    def testing_model(self, data):
        output_list = np.array([])
        for e in data:
            output = self.forward(e)
            output_list = np.append(output_list, output)
        return output_list

    @staticmethod
    def mean_partial_differential(x, mean, stddev):
        return (2 * (x - mean)) / (stddev ** 2)

    @staticmethod
    def stddev_partial_differential(x, mean, stddev):
        return (2 * ((x - mean) ** 2)) / (stddev ** 3)

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

    @property
    def loss_list(self):
        return self.__loss_list

    # @property
    # def mean(self):
    #     return self.mean
    #
    # @property
    # def stddev(self):
    #     return self.stddev
    #
    # @property
    # def weight(self):
    #     return self.weight
