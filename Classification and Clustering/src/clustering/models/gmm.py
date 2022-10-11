import typing as th
from abc import ABCMeta
from sklearn.base import DensityMixin, BaseEstimator

# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish
from sklearn.mixture import GaussianMixture


class GMM(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(
            self,
            cluster_count: int,
            max_iteration: int,
            # add required hyper-parameters (if any)
    ):
        # todo: initialize parameters
        self.n = cluster_count
        self.it = max_iteration
        self.model = GaussianMixture(n_components=self.n, max_iter=self.it)

    def fit(self, x):
        # todo: for you to implement
        xT = x.copy()
        self.model.fit(xT)
        return self

    def predict(self, x):
        # todo: for you to implement
        x1 = x.copy()
        return self.model.predict(x1)
