import numpy as np


class Utils:
    def __init__(self):
        return

    def Sigmoid(self, z):
        return (1/(1+np.exp(-z)))

    def CrossEntropyLoss(self, Y, A, m):
        return np.dot(-1 / m, (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))))

    def CrossEntropyLoss_Optimized(self, A2, Y, parameters):
        m = Y.shape[1] 
        logprobs = np.multiply(np.log(A2),Y)+(1-Y)*(np.log(1-A2))
        cost = -np.sum(logprobs)/m
        cost = np.squeeze(cost)
        
        return cost

    def GradientDescent(self, Hn, learning_rate, dHn):
        return Hn - (learning_rate * dHn)
