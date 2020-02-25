from sklearn import datasets  
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
import scipy as scipy
from scipy.stats import norm
import scipy.stats
from mluem import NaiveBayes

#importing the necessary packages  
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB   

#downloading the iris dataset, splitting it into train set and validation set 
iris = datasets.load_iris()
# print('iris head ==>', iris) 
class_names = iris.target_names
# print('class names ==>', class_names) 

iris_df=pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target']=iris.target

print(iris_df.head())
# print(iris_df.describe())

# X_train, X_test, y_train, y_test = train_test_split(iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']], iris_df['target'], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

skl_NB = GaussianNB()  
skl_NB.fit(X_train, y_train)   
y_predict = skl_NB.predict(X_test)
print(y_predict)  
print("Score skl_NB: {:.2f}".format(skl_NB.score(X_test, y_test)))

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
# print('X_train, X_test, y_train, y_test ==>', X_train, X_test, y_train, y_test)

NB = NaiveBayes()
NB.fit(X_train, y_train)  
y_predict = NB.predict(X_test)
print('y_predict', y_predict)
score = NB.score(y_test, y_predict)
print('score ==>', score)