import typing as th
from sklearn.base import ClusterMixin, BaseEstimator
# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish
from sklearn.cluster import AgglomerativeClustering


class Hierarchical(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            cluster_count: int,
            # add required hyper-parameters (if any)
    ):
        # todo: initialize parameters
        self.n = cluster_count
        self.model = AgglomerativeClustering(n_clusters=self.n)

    def fit_predict(self, x, **kwargs):
        # todo: for you to implement
        xT = x.copy()
        return self.model.fit_predict(xT)
