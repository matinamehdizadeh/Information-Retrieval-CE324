import typing as th  # Literals are available for python>=3.8
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import math

class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            kind,  #: th.Literal['gaussian', 'bernoulli', ],
            # add required hyper-parameters (if any)
    ):
        self.kind = kind
        # todo: initialize parameters
        

    def fit(self, x1, y, **fit_params):
        # todo: for you to implement
        
        x = x1.copy()
        self.m1 = []
        self.s1 = []
        self.m2 = []
        self.s2 = []
        self.k1 = []
        self.k2 = []
        self.prior = (len(y[y == 1])+80) /(len(y)+100) 

        if (self.kind == 'bernoulli'):
            for i in range(256):
                minimum = np.min(x[:, i])
                maximum = np.max(x[:, i])
                x[:, i] = np.where(x[:, i] <= (((maximum - minimum)/2) + minimum), 0, 1)
 
        Px = x[y==1]
        UPx = x[y==0]
        
        for i in range(256):
            self.m1.append(np.mean(Px[:, i]))
            self.s1.append(np.std(Px[:, i]))
            self.k1.append(len(Px[:, i][Px[:, i] == 1])/len(Px))
            self.m2.append(np.mean(UPx[:, i]))
            self.s2.append(np.std(UPx[:, i]))
            self.k2.append(len(UPx[:, i][UPx[:, i] == 1])/len(UPx))

        return self

    def predict(self, x1):
        # todo: for you to implement
        x = x1.copy()
        ep = 0.00001
        p1 = np.log(self.prior + ep)
        p0 = np.log((1-self.prior) + ep)
        if (self.kind == 'bernoulli'):
            minimum = np.min(x)
            maximum = np.max(x)
            x = np.where(x <= (((maximum - minimum)/2) + minimum), 0, 1)
            for i in range(256):
                if (x[i] == 1):
                    p1 += np.log(self.k1[i] + ep)
                    p0 += np.log(self.k2[i] + ep)
                else:
                    p1 += np.log(1-(self.k1[i]) + ep)
                    p0 += np.log(1-(self.k2[i]) + ep)
        else:
            for i in range(256):
                power1 = (-0.5) * math.pow(((x[i] - self.m1[i])/self.s1[i]), 2)
                power2= (-0.5) * math.pow(((x[i] - self.m2[i])/self.s2[i]), 2)
                p1 += np.log(((1/(self.s1[i] * math.sqrt((2*math.pi)))) * math.exp(power1)) + ep)
                p0 += np.log(((1/(self.s2[i] * math.sqrt((2*math.pi)))) * math.exp(power2)) + ep)
        
        if p1 >= p0:
            return 1
        return 0
            