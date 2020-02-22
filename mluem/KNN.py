import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .ModelInterface import ModelInterface
from .Utils import Utils 
#Class Key Nearest Neiborghs
class KNN(ModelInterface):
    def __init__(self, X, Y, k):
        self.X = X
        self.Y = Y
        self.k = k

    def fit(self):
        self.y = (Utils.DistEuclidean (self.X , self.Y))
        return self
   
    def predict(self, X, Y):
        #Classify 
        self.counter = 0
        while len(self.y)>self.k:
            z = self.y.idxmax()
            self.y.pop(z)
            w = self.y.index
            self.Data2Cl = Y.iloc [w,:]
    
        for j in self.Data2Cl.iloc[:,2] ==0:
            self.counter +=j
        if self.counter > len(self.Data2Cl)/2:
            print("Class is CERO")
        else:
            print("Class is ONE")
                
        #Prediction
        Prediction = self.y.mean()

        parameters = {"Euclidean Distance": self.y,
                      "K Nearest Neighbors": self.Data2Cl,
                      "Regression": Prediction}
        #Plot Results
        plt.scatter(self.Data2Cl.iloc[:, 0],self.Data2Cl.iloc[:, 1], marker='*')
        plt.scatter(dato.iloc[:, 0],dato.iloc[:, 1], marker='o')
       
        return parameters
    
    def score(self):  
        return {"Score %":(self.counter/len(self.Data2Cl))*100}