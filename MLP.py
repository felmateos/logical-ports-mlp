import random
import numpy as np

class MultiLayerPerceptron:

    Func = {'tanh' : (lambda x: np.tanh(x))}
    Deriv = {'tanh' : (lambda x: 1-x**2)}
    min_lim = -1
    max_lim = 1

    def __init__(self, params=None):
        if (params == None):
            self.inputLayer = 2
            self.hiddenLayer = 2
            self.outputLayer = 1
            self.learningRate = 0.05
            self.maxEpochs = 12
            self.activationFunc = self.Func['tanh']
            self.activationDeriv = self.Deriv['tanh']

        self.outputs = []
        self.weightsHidden = self.start_weights(self.inputLayer, self.hiddenLayer)
        self.weightsOutput = self.start_weights(self.hiddenLayer, self.outputLayer)
        self.biasHidden = self.start_bias(self.hiddenLayer)
        self.biasOutput = self.start_bias(self.outputLayer)
        self.classesNumber = 3

    def start_weights(self, x, y):
        weights = []
        for _ in range(x):
            tempList = []
            for _ in range(y):
                tempList.append(random.randint(self.min_lim, self.max_lim))
            weights.append(tempList)
        return weights
    
    def start_bias(self, y):
        bias = []
        for _ in range(y):
            bias.append(random.randint(self.min_lim, self.max_lim))
        return bias
    
    def predict(self, x):
        z = self.activationFunc(np.dot(self.weightsHidden, x) + self.biasHidden)
        y = self.activationFunc(np.dot(self.weightsOutput, z) + self.biasOutput)
        return y

    def train(self, x, epochs):
        for _ in range (epochs):
            z = self.activationFunc(np.dot(self.weightsHidden, x) + self.biasHidden)
            y = self.activationFunc(np.dot(self.weightsOutput, z) + self.biasOutput)
        return y
