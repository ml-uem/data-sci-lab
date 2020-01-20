import numpy as np


class Utils:
    def __init__(self):
        return

    def Sigmoid(self, z):
        return (1/(1+np.exp(-z)))

    def CrossEntropyLoss(self, Y, A, m):
        return np.dot(-1 / m, (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))))
