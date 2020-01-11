from LinearRegression import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():

    lineal = LinearRegression()
    df=pd.read_csv('data/1.csv', sep=',',header=None)
    x = df.values[:, 0]
    y = df.values[:, 1]
    plt.scatter(x, y)
    plt.show()
    data = lineal.fit(x,y)
    print(data.slope)
    print(data.intercept)

    #print(lineal.predict(8, x,y))
    print(lineal.predict(8))

    plt.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    main()
