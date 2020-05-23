import numpy as np

class Hungarian():
    def __init__(self):
        self.cost_matrix = []

    def start(self, cost_matrix):
        self.cost_matrix = cost_matrix
        return self.__step1()


    def __step1(self):
        mins = np.amin(self.cost_matrix, axis=1)
        self.cost_matrix = (self.cost_matrix.transpose() - mins).transpose()
        return self.__step2()

    def __step2(self):
        mins = np.amin(self.cost_matrix, axis=0)
        self.cost_matrix = self.cost_matrix - mins
        zeros_cols = np.where(self.cost_matrix.transpose() == 0)[1]
        valid_values = np.unique(zeros_cols)

        ret_val = 0
        if len(valid_values) == len(zeros_cols):
            ret_val = zeros_cols
        else:
            ret_val = self.__step3()
        return ret_val

    def __step3(self):
        return 0



