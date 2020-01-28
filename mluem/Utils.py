import numpy as np


class Utils:    
    def sigmoid(z):
        return 1/ (1 + np.exp(-z))

    def cross_entropy_loss(Y, A, m):
        return np.dot(-1 / m, (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))))        
