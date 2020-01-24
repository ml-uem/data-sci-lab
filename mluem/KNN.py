from .KNN import KNN
import numpy as np
import pandas as pd

class SimpleLinearRegression(ModelInterface):
    def __init__(self):
        self._y = 0
     
    def fit(self, X):
        self._y = (Utils.DistEuclidea (dato , Data2))
    #Classify 
        while len(self._y)>X:
            z = self._y.idxmax()
            self._y.pop(z)

        return self

    def predict(self):
        return (self._y.mean(), self._y)

    def score(self, X, y):
       
