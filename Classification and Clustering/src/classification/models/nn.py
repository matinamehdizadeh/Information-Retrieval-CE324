import typing as th
from sklearn.base import BaseEstimator, ClassifierMixin

# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish
from sklearn.neural_network import MLPClassifier


class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(
            self,mi, hls
            # add required hyper-parameters (if any)
    ):
        # todo: initialize parameters
        self.mi = mi
        self.hls = hls

    def fit(self, x, y, **fit_params):
        # todo: for you to implement
        xT = x.copy()
        yT = y.copy()
        self.model = MLPClassifier(max_iter=self.mi, hidden_layer_sizes= self.hls, activation='identity')
        self.model.fit(xT, yT)
        return self

    def predict(self, x):
        x1 = x.copy()
        return self.model.predict(x1)
        # todo: for you to implement
