import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mluem.SimpleLinearRegression as slr

def main():
    lineal = slr()    
    filepath = '' # add your path ../usecases/simple_linear_regression/data/1.csv
    df = pd.read_csv(filepath, sep=',',header=None)
    x = df.values[:, 0]
    y = df.values[:, 1]

    data = lineal.fit(x,y)
    print(test(y, lineal.predict(x)))
    print('score ==>', lineal.score(x,y))   

    plt.scatter(x, y)
    plt.plot(x,lineal.predict(x))
    plt.show()


def test(y, prediction):
    n = y - prediction
    m = y - np.mean(y)

    return 1 - ( n.dot(n) / m.dot(m) )


if __name__ == "__main__":
    main()
