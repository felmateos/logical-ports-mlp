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
        for _ in range(y): #inverti p facilitar o calculo
            tempList = []
            for _ in range(x): #inverti p facilitar o calculo
                tempList.append(random.uniform(self.min_lim, self.max_lim))
            weights.append(tempList)
        return weights
    
    def start_bias(self, y):
        bias = []
        for _ in range(y):
            bias.append(random.uniform(self.min_lim, self.max_lim))
        return bias
    
    def predict(self, X):
        z = []

        for i in range(self.hiddenLayer):
            zin = 0
            pairs = zip(self.weightsHidden[i], X)
            for w,x in tuple(pairs):
                zin += w*x
            zin += self.biasHidden[i]
            z.append(self.activationFunc(zin))

        y = 0

        for i in range(self.outputLayer):
            yin = 0
            pairs = zip(self.weightsOutput[i], z)
            for w,x in tuple(pairs):
                yin += w*x
            yin += self.biasOutput[i]
            y = self.activationFunc(yin)

            if y >= 0:
                return 1
            else:
                return -1
