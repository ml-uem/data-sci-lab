import numpy as np

class MultipleLinearRegression:
    """
    Multiple Linear Regression Alg
    """
    def __init__(self):
        w = 0

    def fit(self, X, y):

        w = np.linalg.solve(np.dot(X,np.transpose(X)),np.dot(np.transpose(X),y))

    return self

  def predict(self, x):
    """
    """
    return w * x    

  def rsquare(self, x, y):
    n = y - self.predict(x)
    m = y - np.mean(y)
    return 1 - ( n.dot(n) / m.dot(m) )
