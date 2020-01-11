import numpy as np

class LinearRegression:
    """
    Simple Linear regression
    """
    def __init__(self):
        self._slope = 0
        self._intercept = 0
        self._predict = 0

    def fit(self, X, y):
        self._slope = (np.mean(X.dot(y))- (np.mean(X) * np.mean(y))) \
        / (np.mean(X.dot(X)) - np.mean(X)**2)

        self._intercept = ((np.mean(y) * np.mean(X.dot(X))) - np.mean(X) * np.mean(X.dot(y))) \
        / (np.mean(X.dot(X)) - (np.mean(X)**2))

        return self

    def predict(self,X):
        self._predict = X * self._slope + self._intercept
        return self

    def r_square(self,y):
        return 1 - ((np.sum((y - self._predict) ** 2)) / (np.sum((y - np.mean(y)) ** 2)))
