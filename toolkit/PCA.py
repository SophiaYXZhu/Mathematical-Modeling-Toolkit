import numpy as np
import pandas as pd

class PCA:
    def __init__(self, data: pd.DataFrame):
        '''
        data is the matrix containing values of each subject's values to each rating attribute.
        It is a pd.DataFrame() object.
        '''
        self.data = data
        self.eigenvectors = None

    def standardize(self):
        '''
        Returns the standardized data by columns.
        '''
        self.data  = (self.data - self.data.mean()) / (self.data.std())
        return self.data
    
    def covariance(self):
        '''
        Returns the variance-covariance matrix of the data.
        '''
        data = self.data.values
        R, C = data.shape
        cov = np.zeros((C, C))
        for i in range(C):
            miu_i = np.sum(data[:, i]) / R
            for j in range(i, C):
                miu_j = np.sum(data[:, j]) / R
                cov[i, j] = np.sum((data[:, i] - miu_i) * (data[:, j] - miu_j)) / (R - 1)
                if not i == j:
                    cov[j, i] = cov[i, j]
        return cov

    def eigendecomposition(self):
        '''
        Sets the eigenpairs attribute of this PCA instance and returns the contribution of each eigenvalue - eigenvector pair.
        '''
        self.standardize()
        eig_val, eig_vec = np.linalg.eig(self.covariance())
        self.eigenvectors = eig_vec
        contribution = list()
        for i in range(len(eig_val)):
            contribution.append(eig_val[i]/sum(eig_val))
            contribution.sort(reverse=True)
        return contribution

    def get_new_data(self, n):
        '''
        Returns the new data matrix with the first n eigenvectors.
        '''
        self.eigendecomposition()
        eigenmatrix = self.eigenvectors[:,:n]
        new_data = np.matmul(self.data, eigenmatrix)
        return new_data

if __name__ == '__main__':
    data = pd.DataFrame([[90342, 52455, 101091, 19272, 82, 16.1, 197435, 0.172],
        [4903, 1973, 2035, 10313, 34.2, 7.1, 592077, 0.003],
        [6735, 21139, 3767, 1780, 36.1, 8.2, 726396, 0.003],
        [49454, 36241, 81557, 22504, 98.1, 35.9, 348226, 0.985],
        [139190, 203505, 215898, 10609, 93.2, 12.6, 139572, 0.628],
        [12215, 16219, 10351, 6382, 62.5, 8.7, 145818, 0.066], 
        [2372, 6572, 8103, 12329, 184.4, 22.2, 20921, 0.152],
        [11062, 23078, 54935, 23804, 370.4, 41, 65486, 0.263],
        [17111, 23907, 52108, 21796, 221.5, 21.5, 63806, 0.276],
        [1206, 3930, 6126, 15586, 330.4, 29.5, 1840, 0.437],
        [2150, 5704, 6200, 10870, 184.2, 12, 8913, 0.274],
        [5251, 6155, 10383, 16875, 146.4, 27.5, 78796, 0.151],
        [14341, 13203, 19396, 14691, 94.6, 17.8, 6354, 1.574]])
    pca = PCA(data)
    print(pca.eigendecomposition())
