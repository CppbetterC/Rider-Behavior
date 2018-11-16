class Data:

    def __init__(self, *argv):
        dim = []
        for arg in argv:
            dim.append(arg)

        if len(dim) == 7:
            self.date = dim[0]
            self.time = dim[1]
            self.Acc = dim[2]
            self.Gyro = dim[3]
            self.Mag = dim[4]
            self.Pre = dim[5]
            self.Hall = dim[6]

        elif len(dim) == 14:
            self.date = dim[0]
            self.time = dim[1]
            self.acc_x = dim[2]
            self.acc_y = dim[3]
            self.acc_z = dim[4]
            self.gyro_x = dim[5]
            self.gyro_y = dim[6]
            self.gyro_z = dim[7]
            self.mag_x = dim[8]
            self.mag_y = dim[9]
            self.mag_z = dim[10]
            self.pre_x = dim[11]
            self.pre_y = dim[12]
            self.hall = dim[13]

        elif len(dim) == 15:
            self.date = dim[0]
            self.time = dim[1]
            self.acc_x = dim[2]
            self.acc_y = dim[3]
            self.acc_z = dim[4]
            self.gyro_x = dim[5]
            self.gyro_y = dim[6]
            self.gyro_z = dim[7]
            self.mag_x = dim[8]
            self.mag_y = dim[9]
            self.mag_z = dim[10]
            self.pre_x = dim[11]
            self.pre_y = dim[12]
            self.hall = dim[13]
            self.label = dim[14]
        else:
            print('Error Constructor')