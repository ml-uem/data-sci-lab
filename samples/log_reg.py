import numpy as np
import pandas as pd
from mluem import LogisticRegression as LogReg

def main(X, y):   
    model = LogReg(1, 10, 2)
    model.fit(X, y.T)

    prediction = model.predict(X)
    score = model.score(X, y.T)

    print('prediction =>', prediction)
    print('score =>', score)

def load_data():
    filepath = '' # ../usecases/logistic_regression/sample1.csv
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = X.to_numpy()
    y = y.to_numpy().reshape(1,99)
    return X, y

if __name__ == "__main__":  
    X, y = load_data()  
    main(X, y)

  



