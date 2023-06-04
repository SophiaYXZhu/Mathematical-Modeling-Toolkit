import numpy as np
from collections import namedtuple

class AHP:
    def __init__(self, A):
        '''
        A is the comparison matrix in the form of a two dimensional iterable object.
        For example, a case of three comparison subjects with true weights w1, w2, and w3 have the following comparison matrix:
        [[1, w1/w2, w1/w3],
        [w2/w1, 1, w2/w3],
        [w3/w1, w3/w2, 1]].
        '''
        self.A = A
        self.lambda_max = None
    
    def get_weights(self):
        '''
        Get the weights vector of the corresponding comparison matrix (A).
        '''
        lamb,v = np.linalg.eig(self.A) 
        self.lambda_max = max(abs(lamb))
        loc = np.where(lamb == self.lambda_max)
        weight = abs(v[0:len(self.A),loc[0][0]])
        weight = weight/sum(weight)
        return weight
    
    def get_CI(self, RI):
        '''
        Get the consistency ratio of the comparison matrix (A), compared to the idealistic proportions between weights (true comparison matrix).
        You can attain the random indices for number of comparison subjects below 15 here:
        [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.58].
        Returns a named tuple in the following order: (max_eigenvalue, CI, RI, CR).
        '''
        CI = (self.lambda_max-len(self.A))/(len(self.A)-1)
        CR = CI/RI
        ComparisonRatio = namedtuple('ComparisonRatio', ['max_eigenvalue', 'CI', 'RI', 'CR'])
        return ComparisonRatio(self.lambda_max, CI, RI, CR)

if __name__ == '__main__':
    A=[[1, 2, 5], [1/2, 1, 2], [1/5, 1/2, 1]]
    B=[[1, 1/3, 1/8], [3, 1, 1/3], [8, 3, 1]]
    C=[[1, 1, 3], [1, 1, 3], [1/3, 1/3, 1]]
    D=[[1, 3, 4], [1/3, 1, 1], [1/4, 1, 1]]
    E=[[1, 1, 1/4], [1, 1, 1/4], [4, 4, 1]]
    attributes = [[1, 1/2, 4, 3, 3], [2, 1, 7, 5, 5], [1/4, 1/7, 1, 1/2, 1/3], [1/3, 1/5, 2, 1, 1], [1/3, 1/5, 3, 1, 1]]
    arr = [A, B, C, D, E]
    weights_vertical = []
    for i in arr:
        ahp = AHP(i)
        weights_vertical.append(ahp.get_weights())
    subjects_matrix = []
    for c in range(len(weights_vertical[0])):
        tmp = []
        for r in range(len(weights_vertical)):
            tmp.append(weights_vertical[r][c])
        subjects_matrix.append(tmp)
    ahp = AHP(attributes)
    attribute_weights = ahp.get_weights()
    attribute_matrix = [[weight] for weight in attribute_weights]
    print(np.matmul(subjects_matrix, attribute_matrix))