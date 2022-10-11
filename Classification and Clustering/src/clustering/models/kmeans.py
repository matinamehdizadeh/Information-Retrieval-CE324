import typing as th
from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
import random
import numpy as np

class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(
            self,
            cluster_count: int,
            max_iteration: int,
            # add required hyper-parameters (if any)
    ):
        # todo: initialize parameters
        self.n = cluster_count
        self.mi = max_iteration
        self.centroids = [0] * self.n
        

    def fit(self, x):
        xT = x.copy()
        indexes = [0] * len(xT)
        rand = random.sample(range(0, len(xT)), self.n)
        for i in range(self.n):
            self.centroids[i] = xT[rand[i]]
            
        for k in range(self.mi):
            for i in range(len(xT)):
                minimum = 10
                for j in range(self.n):
                    dist = 1 - np.dot(self.centroids[j], xT[i]) / np.sqrt(np.dot(self.centroids[j], self.centroids[j]) * np.dot(xT[i], xT[i]))
                    if dist < minimum:
                        minimum = dist
                        index = j
                indexes[i] = index
            indexes = np.array(indexes)
            for i in range(self.n):
                self.centroids[i] = np.mean(xT[indexes == i], axis=0)
        return self

    def predict(self, x):
        # todo: for you to implement
        x1 = x.copy()
        minimum = 10
        for i in range(self.n):
            dist = 1 - np.dot(self.centroids[i], x1)/np.sqrt(np.dot(self.centroids[i], self.centroids[i])*np.dot(x1, x1))
            if dist < minimum:
                minimum = dist
                index = i
        return index
