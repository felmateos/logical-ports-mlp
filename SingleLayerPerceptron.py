import pandas as pd
import numpy as np

class SingleLayerPerceptron:

    def __init__(self, w, b, a=.5):
        self.w = w
        self.b = b
        self.a = a

    def receive(self, x):
        yin = np.dot(self.w, x) + self.b
        return yin

    def activation(self, yin):
        y = 1 if yin > 0 else 0
        return y

    
    def neuron_process(self, x):
        yin = self.receive(self, x)
        return self.activation(self, yin)
    
    def training(self, x, t):
        y_ = self.neuron_process(self, x)
        s = t - y_

        self.w = [W + X * self.a * s for (W,X) in zip(self.w, x)]

        self.b += self.a * s

        error = np.abs(s)

        return error