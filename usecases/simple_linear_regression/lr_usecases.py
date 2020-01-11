from LinearRegression import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def main():

    lineal = LinearRegression()
    df=pd.read_csv('data/8.csv', sep=',',header=None)
    x = df.values[:, 0]
    y = df.values[:, 1]
    data = lineal.fit(x,y)
    lineal.predict(x)
    r_square = lineal.r_square(8,y)
    print(r_square)


    # plt.scatter(x, y)
    # plt.plot(x,lineal.predict(x))
    # plt.show()


if __name__ == "__main__":
    main()
