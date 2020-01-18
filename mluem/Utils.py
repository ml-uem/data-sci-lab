import numpy as np


class Utils:
    def Sigmoid(self,z):
        return (1/(1+np.exp(-z)))
