import numpy as np

class LinearRegression:
  """
    Simple Linnear regression
  """
  def __init__(self):
    self._slope = 0
    self._intercept = 0

  def fit(self, X, y):
    """

    """
    self._slope = (np.mean(X.dot(y))- (np.mean(X) * np.mean(y))) \
    / (np.mean(X.dot(X)) - np.mean(X)**2)

    self._intercept = ((np.mean(y) * np.mean(X.dot(X))) - np.mean(X) * np.mean(X.dot(y))) \
    / (np.mean(X.dot(X)) - (np.mean(X)**2))

    return self

  def predict(self, x):
    """
    """
    return x * self._slope + self._intercept
