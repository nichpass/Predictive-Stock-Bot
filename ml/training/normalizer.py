import numpy as np
class Normalizer():

    def __init__(self):
        self.mu = None
        self.sd = None


    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0))
        self.sd = np.std(x, axis=(0))
        normalized_x = (x - self.mu)/self.sd
        return normalized_x


    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

    
    def fit_transform(x):
        mu = np.mean(x, axis=(0))
        sd = np.std(x, axis=(0))
        normalized_x = (x - mu) / sd
        return normalized_x
