import numpy as np


class Hungarian():
    def __init__(self):
        self.original_cost_mat = []
        self.cost_matrix = []
        self.int_infinity = 99999999999

    def start(self, cost_matrix):
        self.cost_matrix = cost_matrix
        self.original_cost_mat = cost_matrix.copy()
        return self.__step1()

    def __step1(self):
        mins = np.amin(self.cost_matrix, axis=1)
        self.cost_matrix = (self.cost_matrix.transpose() - mins).transpose()
        return self.__step2()

    def __step2(self):
        mins = np.amin(self.cost_matrix, axis=0)
        self.cost_matrix = self.cost_matrix - mins
        zeros_cols = np.where(self.cost_matrix.transpose() == 0)
        zeros_cols = zeros_cols[1]
        valid_values = np.unique(zeros_cols)

        return self.__step3(valid_values, zeros_cols)


    def __step3(self, valid, zeros):
        if len(valid) == len(zeros):
            ret_val = (zeros, self.__get_total_cost(zeros))
        else:
            ret_val = self.__step4(zeros)
        return ret_val

    def __step4(self, zeros_cols):
        current_cost_mat = self.cost_matrix.copy()
        pairs = []
        already = []
        for ind, val in enumerate(zeros_cols):
            if already.count(val) == 1:
                pairs.append((ind, val))
            already.append(val)
        for val in pairs:
            current_cost_mat[val[1]][:] = self.int_infinity
        zeros = np.unique(np.where(current_cost_mat == 0)[1])
        for val in zeros:
            current_cost_mat[:, val] = self.int_infinity
        minimum = np.amin(current_cost_mat)
        current_cost_mat -= minimum
        for i in range(current_cost_mat.shape[0]):
            for j in range(current_cost_mat.shape[1]):
                if current_cost_mat[i][j] < self.int_infinity - minimum:
                    self.cost_matrix[i][j] = current_cost_mat[i][j]
        return self.__step5()

    def __step5(self):
        result = np.count_nonzero(self.cost_matrix == 0, axis=0)
        aux = result.copy()
        associations = []
        for i in range(len(result)):
            aux_arg_min = np.argmin(aux)
            result_arg_min = np.where(result == aux[aux_arg_min])[0]
            aux = np.delete(aux, aux_arg_min)
            zero_pos = np.where(self.cost_matrix[:, result_arg_min] == 0)[0]
            for j in range(len(zero_pos)):
                val = zero_pos[j]
                if associations.count(val) == 0:
                    associations.append(val)
                    break
        return associations, self.__get_total_cost(associations)

    def __get_total_cost(self, associations):
        total = 0
        for i, val in enumerate(associations):
            total += self.original_cost_mat[i][val]
        return total
