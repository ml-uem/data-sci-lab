import numpy as np
from ./ModelInterface import ModelInterface

class MultipleLinearRegression(ModelInterface):
    """
    Multiple Linear regression
    """
    def __init__(self):
        self._slope = 0        

    def fit(self, X, y):
        self._slope = np.linalg.solve(np.transpose(X).dot(X), np.transpose(X).dot(y))
        return self


    def predict(self, x):    
        return x * self._slope

