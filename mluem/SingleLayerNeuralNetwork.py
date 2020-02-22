from .ModelInterface import ModelInterface
from .Utils import Utils
import numpy as np


class SingleLayerNeuralNetwork(ModelInterface):

    def __init__(self, n_h, num_iterations=10000, learning_rate=0.1, print_cost=False):
        self._n_h = n_h
        self._num_iterations = num_iterations
        self._learning_rate = learning_rate
        self._print_cost = print_cost

        self._m = None
        self._parameters = None

    def _propagation(self, X, Y):
        Z1 = np.dot(self._parameters["W1"], X) + self._parameters["b1"]
        A1 = np.tanh(Z1)
        Z2 = np.dot(self._parameters["W2"], A1) + self._parameters["b2"]
        A2 = (1/(1+np.exp(-Z2)))

        W2 = self._parameters["W2"]
        dZ2 = A2 - Y
        dW2 = (1/self._m)*np.dot(dZ2, A1.T)
        dB2 = (1/self._m)*np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(W2.T * dZ2, 1-np.power(A1, 2))
        dW1 = (1/self._m)*np.dot(dZ1, X.T)
        dB1 = np.sum(dZ1, axis=1, keepdims=True)

        return A2, {"dW2": dW2, "dB2": dB2, "dW1": dW1, "dB1": dB1}

    def fit(self, X, y):
        np.random.seed(3)
        self._m = y.shape[1]

        # layer_sizes
        n_x = X.shape[0]
        n_h = self._n_h
        n_y = y.shape[0]

        # initialize_parameters
        W1 = np.random.randn(n_h, n_x)*0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h)*0.01
        b2 = np.zeros((n_y, 1))
        self._parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

        for i in range(0, self._num_iterations):
            A2, grads = self._propagation(X, y)
            cost = Utils().CrossEntropyLoss_Optimized(A2, y, self._parameters)
            W1 = self._parameters["W1"] - (self._learning_rate * grads["dW1"])
            W2 = self._parameters["W2"] - (self._learning_rate * grads["dW2"])
            b1 = self._parameters["b1"] - (self._learning_rate * grads["dB1"])
            b2 = self._parameters["b2"] - (self._learning_rate * grads["dB2"])

            self._parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

            if self._print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return self._parameters

    def predict(self, X):
        z1 = np.dot(self._parameters["W1"], X) + self._parameters["b1"]
        a1 = np.tanh(z1)
        z2 = np.dot(self._parameters["W2"], a1) + self._parameters["b2"]
        a2 = (1/(1+np.exp(-z2)))

        self._predictions = (a2 > 0.5)

        return self._predictions

    def score(self, X, y):
        return np.mean(y == self.predict(X).astype(int))
