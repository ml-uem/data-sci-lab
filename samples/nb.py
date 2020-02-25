import matplotlib.pyplot as plt
import numpy as np
from mluem import NaiveBayes  
from sklearn import datasets  
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB   

#downloading the iris dataset
iris = datasets.load_iris()
# print('iris head ==>', iris) 
class_names = iris.target_names

# The indices of the features that we are plotting
x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()

print('class names ==>', class_names)
print("class 0: {0}, class 1: {1}, class 2: {2}".format(class_names[0],class_names[1], class_names[2]))

# splitting it into train set and validation set 
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

# sklearn implementation
skl_NB = GaussianNB()  
skl_NB.fit(X_train, y_train)   
y_predict = skl_NB.predict(X_test)
print('y_predict', y_predict) 
print("Score skl_NB: {:.2f}".format(skl_NB.score(X_test, y_test)))

# mluem implementation
NB = NaiveBayes()
NB.fit(X_train, y_train)  
y_predict = NB.predict(X_test)
print('y_predict', y_predict)
score = NB.score(y_test, y_predict)
print('score NB ==>', score)

plt.show()