from collections import namedtuple
import pandas as pd
import numpy as np

class TOPSIS:
    def __init__(self, data):
        '''
        data is the matrix containing values of each subject's values to each rating attribute.
        It is a np.array() object.
        '''
        self.data = data
    
    def convert(self, scale):
        '''
        This method converts the columns of data large indices based on the scale array.
        The scale array should indicate whether each column is a large ('l'), small ('s'), or centered ('c') index.
        The method returns self.data and changes self.data.
        '''
        assert len(scale) == len(self.data[0])
        for c in range(self.data.shape[1]):
            ma = max(self.data[:, c])
            mi = min(self.data[:, c])
            if scale[c] == 'c':
                for x in range(self.data.shape[0]):
                    self.data[x, c] = 2(self.data[x, c] - mi)/(ma-mi) if mi <= self.data[x, c] <= (ma+mi)//2 else 2(ma - self.data[x, c])/(ma-mi)
            elif scale[c] == 's':
                self.data[:, c] = 1/self.data[:, c]
            # normalize
            self.data[:, c] = self.data[:, c] / np.sqrt(sum(self.data[:, c]**2))
        return self.data

    def compute_dist(self):
        '''
        Returns the distance between each subject of evaluation and the ranking of each subject as lists.
        '''
        mi = [min(self.data[:, c]) for c in range(self.data.shape[1])]
        ma = [max(self.data[:, c]) for c in range(self.data.shape[1])]
        dist = [None * self.data.shape[0]]
        for r in range(self.data.shape[0]):
            D_pos = 0
            D_neg = 0
            for c in range(self.data.shape[1]):
                D_pos += (self.data.shape[r, c] - ma[c])**2
                D_neg += (self.data.shape[r, c] - mi[c])**2
            dist[r] = D_neg/(D_pos + D_neg)
        Distance = namedtuple('Distance', ['distance', 'ranking'])
        ordered_dist = [(dist[i], i) for i in range(len(dist))]
        ordered_dist.sort(value = lambda x: x[0])
        ranking = [ordered_dist(i)+1 for i in range(ordered_dist)]
        return Distance(dist[r], ranking)