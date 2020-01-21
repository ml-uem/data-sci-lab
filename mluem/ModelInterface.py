import numpy as np

class ModelInterface:
    def __init__(self): raise NotImplementedError

    def fit(self, X, y): raise NotImplementedError

    def predict(self, X): raise NotImplementedError

    def score(self, X, y): raise NotImplementedError
