from LinearRegression import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():

    lineal = LinearRegression()
    df=pd.read_csv('data/1.csv', sep=',',header=None)
    x = df.values[:, 0]
    y = df.values[:, 1]
    data = lineal.fit(x,y)
    print(data._slope)
    print(data._intercept)

    print(lineal.predict(8))

    plt.scatter(x, y)
    plt.plot(x,lineal.predict(x))
    plt.show()


if __name__ == "__main__":
    main()
