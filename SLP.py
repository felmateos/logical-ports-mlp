import numpy as np

class Neuron:

    def __init__(self, weights, bias, learning_rate=.5):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

    def receive(self, inputs):
        answer = np.dot(self.weights, inputs)
        answer += self.bias
        return answer
    
    def activation(self, answer):
        propagate = 1 if answer > 0 else -1
        return propagate
    
    def process(self, inputs):
        answer = self.receive(inputs)
        return self.activation(answer)
    
    def learn(self, inputs, target):
        result = self.process(inputs)

        diff = target - result
        
        self.weights = [W + X * self.learning_rate * diff for (W,X) in zip(self.weights, inputs)]

        self.bias += self.learning_rate * diff

        error = np.abs(diff)

        return error
