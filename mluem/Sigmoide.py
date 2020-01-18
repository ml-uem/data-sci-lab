#Funcion Sigmoide
import math
#import pandas as pd
import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt
import math
def sigmoide (z1):
    x=1/(1+ math.exp(-z1))
    return x
#Check and plot
#for z1 in range(-10,10,1):#range(10):[-10,10]
#    print(sigmoide(z1))
#    plt.scatter(z1,sigmoide(z1), color='orange')
    