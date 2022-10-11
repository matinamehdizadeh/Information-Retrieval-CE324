import typing as th
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy import spatial

class KNN(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            #k: int,
            # add required hyper-parameters (if any)
    ):
        pass

    def fit(self, x, y, **fit_params):
        # todo: for you to implement
        self.xT = x.copy()
        self.yT = y.copy()
        
        return self

    def predict(self, x, k):
        # todo: for you to implement
        x1 = x.copy()
        near = []
        p1 = 0
        p0 = 0
        for i in range(0, len(self.xT), 2):
            dist = 1 - spatial.distance.cosine(x1, self.xT[i])
            near.append([dist, self.yT[i]])
        near =  sorted(near, key=lambda x: x[0])
        #print(near)
        for i in range(k):
            if near[i][1] == 1:
                p1+=1
            else:
                p0+=1
        if p1>p0:
            return 1
        return 0
