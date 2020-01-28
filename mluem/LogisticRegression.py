import numpy as np
from .ModelInterface import ModelInterface
from .Utils import Utils 

class LogisticRegression(ModelInterface):

    def __init__(self, learning_rate, epochs, print_iter):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._print_iter = print_iter

    def _initialize(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = 0
        self.m = dim
        return self

    def _propagation(self, X, Y):
        # forward propagation
        activation = Utils.sigmoid(np.dot(self.w.T, X) + self.b)
        cost = Utils.cross_entropy_loss(Y, activation, self.m)

        # backward propagation
        dw = 1 / self.m * np.dot(X, (activation - Y).T)
        db = np.dot(1 / self.m, np.sum(activation - Y))

        cost = np.squeeze(cost)

        return activation, cost, dw, db

    def fit(self, X, Y):
        self._initialize(X.shape[0])

        for i in range(self._epochs):
            activation, cost, dw, db = self._propagation(X, Y)

            self.w = self.w - self._learning_rate * dw
            self.b = self.b - self._learning_rate * db

            if (self._print_iter and i % self._epochs == 0):
                print('the epoch is: %i the error is %f' %(i, cost))


    def predict(self, X):
        Y_prediction = np.zeros((1, self.m))

        activation = Utils.sigmoid(np.dot(self.w.T, X) + self.b) 

        for i in range(activation.shape[1]):
            if activation[0][i] <= 0.5:
                Y_prediction[0][i] = 0        
            else:
                Y_prediction[0][i] = 1  

        return Y_prediction        

    def score(self, X, Y):    
        return np.mean(Y == self.predict(X))

