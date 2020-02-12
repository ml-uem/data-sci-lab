from .ModelInterface import ModelInterface
import numpy as np

class MultipleLinearRegression(ModelInterface):
    def __init__(self):
        self._w = 0

    def fit(self, X, y):
        self._w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        return self

    def predict(self, X):
        return self._w * X

    def score(self, X, y):
        m = y - np.mean(y)
        n = y - self.predict(X)
        return 1 - (n.dot(n) / m.dot(m))    
