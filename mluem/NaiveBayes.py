import numpy as np
import math
from .ModelInterface import ModelInterface

class NaiveBayes(ModelInterface):
    def __init__(self):
        self.classes = []
        self.class_freq = {}
        self.means = {}
        self.std = {}
        self.class_prob = {}

    def separate_by_classes(self, X, y):
        self.classes = np.unique(y)        
        classes_index = {}
        subdatasets = {}
        klass, counts = np.unique(y, return_counts=True)        
        self.class_freq = dict(zip(klass, counts))
        for class_type in self.classes:
            classes_index[class_type] = np.argwhere(y==class_type)
            subdatasets[class_type] = X[classes_index[class_type], :]
            self.class_freq[class_type] = self.class_freq[class_type]/sum(list(self.class_freq.values()))
        return subdatasets
    
    def fit(self, X, y):
        separated_X = self.separate_by_classes(X, y)        
        for class_type in self.classes:
            # Here we calculate the mean and the standart deviation from datasets
            self.means[class_type] = np.mean(separated_X[class_type], axis=0)[0]
            self.std[class_type] = np.std(separated_X[class_type], axis=0)[0]
    
    def calculate_probability(self, x, mean, stdev):
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def predict_proba(self, X):
        self.class_prob = {klass:math.log(self.class_freq[klass], math.e) for klass in self.classes}
        for klass in self.classes:
            for i in range(len(self.means)):
                self.class_prob[klass]+=math.log(self.calculate_probability(X[i], self.means[klass][i], self.std[klass][i]), math.e)
        self.class_prob = {klass: math.e**self.class_prob[klass] for klass in self.class_prob}
        return self.class_prob

    def predict(self, X):        
        pred = []
        for x in X:
            pred_class = None
            max_prob = 0
            for klass, prob in self.predict_proba(x).items():
                if prob>max_prob:
                    max_prob = prob
                    pred_class = klass
            pred.append(pred_class)
        return np.asarray(pred)

    def score(self, y_test, y_predict):
        correct = 0
        for i in range(len(y_test)):
            if y_test[i] == y_predict[i]:
                correct += 1
        return correct / float(len(y_test))