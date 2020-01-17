from .ModelInterface import ModelInterface

import numpy as np


class MultipleLinearRegression(ModelInterface):
    def __init__(self):
        self._w = 0

    def fit(self, X, y):
        self._w = np.linalg.solve(
            np.dot(X, np.transpose(X)), np.dot(np.transpose(X), y))
        return self

    def predict(self, X):
        return self._w * X

    def rsquare(self, X, y):
        return 1 - (n.dot(y - self.predict(X)) / m.dot(y - np.mean(y)))
