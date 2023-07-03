import random
import numpy as np
import pandas as pd

class MultiLayerPerceptron:

    Act = {'tanh' : (lambda x: np.tanh(x))}
    DerivA = {'tanh' : (lambda x: 1-x**2)}
    Error = {'SR' : (lambda x,y: (x-y)**2)}
    DerivE = {'SR' : (lambda x,y: -2*(x-y))}
    min_lim = -1
    max_lim = 1

    def __init__(self, params=None):
        if (params == None):
            self.L0 = 2
            self.L1 = 2
            self.L2 = 1
            self.learning_rate = 0.1
            self.max_epochs = 12
            self.activation_func = self.Act['tanh']
            self.activation_deriv = self.DerivA['tanh']
            self.error_func = self.Error['SR']
            self.error_deriv = self.DerivE['SR']

        self.outputs = []
        self.weights_L1 = self.start_weights(self.L0, self.L1)
        self.weights_L2 = self.start_weights(self.L1, self.L2)
        self.bias_L1 = self.start_bias(self.L1)
        self.bias_L2 = self.start_bias(self.L2)
        self.classes_number = 2

    def start_weights(self, i, j):
        weights = []
        for _ in range(j): #inverti p facilitar o calculo
            tempList = []
            for _ in range(i): #inverti p facilitar o calculo
                tempList.append(random.uniform(self.min_lim, self.max_lim))
            weights.append(tempList)
        return np.array(weights)
    
    def start_bias(self, j):
        bias = []
        for _ in range(j):
            bias.append(random.uniform(self.min_lim, self.max_lim))
        return np.array(bias)
    
    def feed_forward(self, x):
        self.output_L1 = []
        self.output_L2 = []

        temp = []

        for i,w in enumerate(self.weights_L1):
            temp = [X*W for (X, W) in zip(x, w)]
            temp = np.sum(temp) + self.bias_L1[i]

        self.output_L1.append(self.activation_func(temp))
        self.output_L1 = np.array(self.output_L1)

        temp = []

        for i,w in enumerate(self.weights_L2):
            temp = [X*W for (X, W) in zip(x, w)]
            temp = np.sum(temp) + self.bias_L2[i]

        self.output_L2.append(self.activation_func(temp))
        self.output_L2 = np.array(self.output_L2)

    def one_hot_encoding(self, y):
        classes = pd.unique(y)
        classes = np.sort(classes)
        Y = []
        for v in y:
            temp = np.zeros(len(self.L2))
            temp[np.where(classes == v)[0]] = 1
            Y.append()
        return np.array(Y)

    def error_calculation(self, y):
        total_error = 0
        for i in range(self.L2):
            SR = self.error_func(y[i], self.output_L2[i])
            total_error += SR
        return total_error        

    def back_propagation(self, x, y):
        # gradient descent L2
        delta_L2 = self.error_deriv(y, self.output_L2) * self.activation_deriv(self.output_L2)

        # updating weights and biases L2
        for i in range(self.L2):
            for j in range(self.L1):
                self.weights_L2[i][j] -= self.learning_rate * delta_L2[i] * self.output_L1[j]
                self.bias_L2 -= self.learning_rate * delta_L2[i]

        # gradient descent L1
        delta_L1 = []


    
    def fit(self, X, y):
        epoch = 0

        while epoch < self.max_epochs:
            for i, r in X.iterrows():
                x = [r[c] for c in X.columns]
                
                # feed forward
                self.feed_forward(x)

                # one-hot encoding
                Y = self.one_hot_encoding(y)

                # error calculation
                total_error = self.error_calculation(Y[i])

                # back propagation
                self.back_propagation(x, Y[i])