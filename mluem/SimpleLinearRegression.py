from .ModelInterface import ModelInterface

import numpy as np


class SimpleLinearRegression(ModelInterface):
    def __init__(self):
        self._slope = 0
        self._intercept = 0

    def fit(self, X, y):
        self._slope = (np.mean(X.dot(y)) - (np.mean(X) * np.mean(y))) \
            / (np.mean(X.dot(X)) - np.mean(X)**2)

        self._intercept = ((np.mean(y) * np.mean(X.dot(X))) - np.mean(X) * np.mean(X.dot(y))) \
            / (np.mean(X.dot(X)) - (np.mean(X)**2))

        return self

    def predict(self, X):
        return X * self._slope + self._intercept

    def rsquare(self, X, y):
        n = y - self.predict(X)
        m = y - np.mean(y)
        return 1 - (n.dot(n) / m.dot(m))
