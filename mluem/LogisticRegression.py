import numpy as np
from .ModelInterface import ModelInterface
from .Utils import Utils 

class LogisticRegression(ModelInterface):

  def __init__(self, learning_rate, epoch, print_iter):
    self._learning_rate = learning_rate
    self._epoch = epoch
    self._print_iter = print_iter
    return self    

  def _initialize(self, dim):
    self.w = np.zeros((dim, 1))
    self.b = 0
    self.m = dim
    return self

  def _propagation(X, Y):
    # forward propagation
    activation = Utils.Sigmoid(np.dot(self.w.T, X) + self.b)
    cost = Utils.CrossEntropyLoss(Y, activation, self.m)

    # backward propagation
    dw = np.dot(1/m, np.dot(X, (activation-Y).T))
    db = np.dot(1/m, np.sum(activation-Y))

    cost = np.squeeze(cost)

    return activation, cost, dw, db

  def fit(X, Y):
    self._initialize(X.shape(0))
    print(self.w.T.shape)

    i = 0

    while self.epoch <= i:
      activation, cost, dw, db = self.propagation(X, Y)

      self.w = self.w - self.learning_rate * dw
      self.b = self.b - self.learning_rate * db

      if (self._print_iter % i == 0):
        print('the epoch is: ' + i + ' the error is: ' + self.cost)
      
      i += 1

    print(self.w.T.shape)

  def predict(X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    activation = Utils.Sigmoid(np.dot(self.w.T, X) + self.b)    

    for i in range(activation.shape[1]):
      if (activation >= 0.5):
        Y_prediction[0][i] = 1        
      else:
        Y_prediction[0][i] = 0  

    return Y_prediction        

  def score(self, X, Y):    
    return np.mean(Y == self.predict(X))

