import numpy as np


class Utils:
    def Sigmoid(self,z):
        return (1/(1+np.exp(-z)))

    def CrossEntropy(self, a,y):
        return np.mean(y * np.log(a) + (1-y) * np.log(1-a))
    #Euclidean Distance function
    def DistEuclidean (self,dato,dato1):
           x=0
           for i in range(len(dato1.columns)-1):
               x += (((dato.iloc[0,i] - dato1.iloc[:,i])**2))
               return np.sqrt(x)

    def CrossEntropyLoss_Optimized(self, A2, Y, parameters):
        m = Y.shape[1] 
        logprobs = np.multiply(np.log(A2),Y)+(1-Y)*(np.log(1-A2))
        cost = -np.sum(logprobs)/m
        cost = np.squeeze(cost)
        
        return cost

    def GradientDescent(self, Hn, learning_rate, dHn):
        return Hn - (learning_rate * dHn)

    def compute_cost(self, A2, Y, parameters):
        m = Y.shape[1] 
        logprobs = np.multiply(np.log(A2),Y)+(1-Y)*(np.log(1-A2))
        cost = -np.sum(logprobs)/m
        cost = np.squeeze(cost)
        return cost
