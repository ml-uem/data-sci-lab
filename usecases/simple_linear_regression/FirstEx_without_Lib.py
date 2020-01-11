import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import LinearRegression as lr

def Linear_regression (dataset):
    linealModel = lr.LinearRegression()

    X = dataset[:,0]
    y = dataset[:,1]

    linealModel.fit(X, y)

    my_prediction = linealModel.predict(X)

    return X, y, my_prediction    

def plotPrediction(X, y, prediction):
    plt.scatter(X, y)
    plt.plot(X,prediction)
    plt.show()

def r_square(y, prediction):
    ''' 
        Representa Lo bien que se aproxima a la nube 
        de puntos en comparacion a la media
    '''
    n = y - prediction
    m = y - np.mean(y)
    rSquare = 1 - ( n.dot(n) / m.dot(m) )

    print(f"R square = {rSquare}")

def main():
    
    # ----- Select name of file and read it ----- #
    nameFile = 1

    firstCSV = f"data/{nameFile}.csv"
    dataframe1 = pd.read_csv(firstCSV, sep=",")

    # ----- Convert pandas data frame to a numpy array ----- #
    dataset1 = dataframe1.to_numpy()
    print(f"Dataset we are working with: {dataset1}")

    # ----- Obtain prediction ----- #
    X, y, prediction = Linear_regression(dataset1)

    # ----- Plot vector and prediction ----- #
    plotPrediction(X, y, prediction)

    # ----- Point cloud approach compared to the average ----- #
    r_square(y, prediction)

if __name__ == "__main__":
    main()

