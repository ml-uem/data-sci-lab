import numpy as np


class Utils:
    def Sigmoid(self,z):
        return (1/(1+np.exp(-z)))

    def CrossEntropy(self, a,y):
        return np.mean(y * np.log(a) + (1-y) * np.log(1-a))

    def DistEuclidea (self,dato,dato1):
        x=0
        for i in range(len(dato1.columns)-1):
            x += (((dato.iloc[0,i] - dato1.iloc[:,i])**2))
        return np.sqrt(x)