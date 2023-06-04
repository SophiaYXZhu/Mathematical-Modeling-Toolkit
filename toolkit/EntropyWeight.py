import numpy as np

class EntropyWeight:
    def __init__(self, data):
        '''
        data is the matrix containing values of each subject's values to each rating attribute.
        It is a np.array() object.
        '''
        self.data = data
    
    def get_weights(self):
        '''
        Returns the completely objective weights based on information entropy of each column as an array.
        '''
        for c in range(self.data.shape[1]):
            mi = min(self.data[:, c])
            ma = max(self.data[:, c])
            self.data[:, c] = (self.data[:, c] - mi)/(ma-mi)
        ds = [0 for j in range(self.data.shape[1])]
        for c in range(self.data.shape[1]):
            p = self.data[:, c]/sum(self.data[:, c])
            sum_entropy = 0
            for i in range(len(p)):
                if p[i] > 0.0001:
                    sum_entropy += p[i]*np.log(p[i])
            e = -(1/np.log(self.data.shape[0])) * sum_entropy
            d = 1 - e
            ds[c] = d
        weights = [0 for j in range(self.data.shape[1])]
        sum_d = 0
        for i in ds:
            sum_d += i
        for c in range(len(ds)):
            weights[c] = ds[c]/sum_d
        return weights