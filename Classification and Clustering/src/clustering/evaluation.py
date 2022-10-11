import typing as th
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

def purity(y, y_hat) -> float:
    # todo: for you to implement
    count = 0
    n = np.unique(y_hat)
    for i in n:
        yP = y[y_hat == i]
        count += np.max(np.bincount(yP))
    return count/len(y)


def adjusted_rand_index(y, y_hat) -> float:
    # todo: for you to implement
    return adjusted_rand_score(y, y_hat)



evaluation_functions = dict(purity=purity, adjusted_rand_index=adjusted_rand_index)


def evaluate(y, y_hat) -> th.Dict[str, float]:
    return {name: func(y, y_hat) for name, func in evaluation_functions.items()}