import numpy as np


class Utils:    
    def Sigmoid(z):
        return (1/(1+np.exp(-z)))

    def CrossEntropyLoss(Y, A, m):
        m = Y.shape[1] 
        logprobs = np.multiply(np.log(A),Y)+(1-Y)*(np.log(1-A))
        cost = -np.sum(logprobs)/m
        cost = np.squeeze(cost)

        return cost
