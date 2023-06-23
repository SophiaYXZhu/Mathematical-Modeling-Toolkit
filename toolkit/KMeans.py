import numpy as np
import random

class KMeansNew:
    def __init__(self, num_clusters: int, data):
        '''
        Data should be an iterable of iterables (points).
        '''
        assert len(data) > num_clusters, "The length of your data should be greater than the number of clusters"
        self.num_clusters = num_clusters
        self.data = data
        self.centroids = [-1] * self.num_clusters
        self.clusters = {} # idx: data points
    
    def __dist(self, p1, p2):
        assert len(p1) == len(p2)
        dist = 0
        for i in range(len(p1)):
            dist += (p1[i] - p2[i])**2
        return np.sqrt(dist)

    def __select_points(self):
        chosen_points = []
        for i in range(self.num_clusters):
            chosen_idx = random.randint(0, len(self.data) - 1)
            while chosen_idx in chosen_points:
                chosen_idx = random.randint(0, len(self.data) - 1)
            self.centroids[i] = self.data[chosen_idx]
            chosen_points.append(chosen_idx)
        for i in range(self.num_clusters):
            self.clusters[i] = [self.centroids[i]]
        
    def __form_clusters(self):
        for i in range(len(self.data)):
            if self.data[i] not in self.centroids:
                distances = {} # key: index of centroid / value: distance between that centroid and the data point
                for j in range(self.num_clusters):
                    distances[j] = self.__dist(self.data[i], self.centroids[j])
                distances = sorted(distances.items(), key=lambda x:x[1])
                for k in range(self.num_clusters):
                    if self.data[i] in self.clusters[k]:
                        self.clusters[k].remove(self.data[i])
                self.clusters[distances[0][0]].append(self.data[i])

    def __compute_centroid(self):
        new_centroids = []
        for (k, v) in self.clusters.items():
            sum = np.array([0]*len(self.data[0]))
            for p in v:
                sum += np.array(p)
            sum = sum/len(v)
            new_centroids.append(sum.tolist())
        return new_centroids
    
    def __revise_cluster(self):
        self.__select_points()
        self.__form_clusters()
        new_centroids = self.__compute_centroid()
        while new_centroids != self.centroids:
            self.centroids = new_centroids 
            self.__form_clusters()
            self.__compute_centroid

    def get_cluster(self):
        self.__revise_cluster()
        return self.clusters
    

if __name__ == '__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt #加载数据集，是一个字典类似中的Javamap
    from sklearn.cluster import KMeans
    import pandas as pd
    import numpy as np
    lris_df = datasets.load_iris() #挑选出前两个维度作为横轴和纵轴，你也可以选择其他维度 
    x_axis = lris_df.data[:,0]
    y_axis = lris_df.data[:,2]
    #调试需要分的类别数
    model = KMeans(n_clusters=3) 
    dat = np.array(lris_df.data)

    # sklearn result
    model.fit(dat) #训练模型
    plt.scatter(x_axis, y_axis, c=model.labels_) 
    plt.show() #打印聚类散点图

    # my result
    dat = dat.tolist()
    kmeans = KMeansNew(3, dat)
    cluster = kmeans.get_cluster()
    color_list = ["black", "blue", "yellow"]
    for (k, v) in cluster.items():
        xlist = []
        ylist = []
        for p in v:
            xlist.append(p[0])
            ylist.append(p[2])
        plt.plot(xlist, ylist, color=color_list[k])