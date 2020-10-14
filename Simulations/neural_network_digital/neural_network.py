# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:32:50 2019
NN code adapted from:
    https://thecodacus.com/neural-network-scratch-python-no-libraries/
@author: Simon
"""
import math
import numpy as np


class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0

class Neuron:
    eta = 0.001
    alpha = 0.1

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass  # Input
        else:
            for neuron in layer:  # Otherwise
                con = Connection(neuron)
                self.dendrons.append(con)

    def __repr__(self):
        return "<Neuron a:%s>" % (self.dendrons)

    def __str__(self):
        return "From str method of Neuron: weight is %s" % (1)

    def addError(self, err):
        self.error = self.error + err

###############################################################################
# The nonlinear function is here
###############################################################################
    # def non_linear_activation(self, x):
    #     return 1 / (1 + math.exp(-x * 1.0))

    # def deriv_non_linear_activation(self, x):
    #     return x * (1.0 - x)

    # def non_linear_activation(self, x):
    #     return max(0.0, x)

    # def deriv_non_linear_activation(self, x):
    #     return 1 if x >= 0 else 0

    def non_linear_activation(self, x):
        return x

    def deriv_non_linear_activation(self, x):
        return 1

    # def sigmoid(self, x):
    # def non_linear_activation(self, x):
    #     return 1 / (1 + math.exp(-x * 1.0))

    # # def dSigmoid(self, x):
    # def deriv_non_linear_activation(self, x):
    #     return x * (1.0 - x)
###############################################################################
# End of the non-linear activation function
###############################################################################

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return (self.output)

    def feedForward(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOutput += dendron.connectedNeuron.getOutput() * dendron.weight
        self.output = self.non_linear_activation(sumOutput)

    def getWeights(self):
        dendron_weights = []
        for dendron in self.dendrons:
            dendron_weights.append(dendron.weight)
        return dendron_weights

    def backPropagate(self):
        self.gradient = self.error * self.\
            deriv_non_linear_activation(self.output)
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta*(
                    dendron.connectedNeuron.output*self.gradient) + \
                        self.alpha*dendron.dWeight
            dendron.weight = dendron.weight + dendron.dWeight
            dendron.connectedNeuron.addError(dendron.weight * self.gradient)
        self.error = 0


class Network:
    def __init__(self, topology, biasing):
        self.layers = []
        self.biasing = biasing
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    # This is setting the input neurons
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            if self.biasing:
                layer.append(Neuron(None))  # This is the bias neuron
                layer[-1].setOutput(1)  # Set biasing to 1, weight does rest
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def feedForward(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForward()

    def getWeights(self):
        weights = []
        for layer in self.layers[1:]:
            for neuron in layer:
                weights.append(neuron.getWeights())
        return weights

    def backPropagate(self, target):  # This loooks goods
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] -
                                        self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()

    def getError(self, target):  # Will change this probably
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def getResults(self):  # Might change this?
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        if self.biasing:
            output.pop()
        return output

    def getThResults(self):  # Sould change this, get_threshold_results
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output = [1 if x == max(output) else 0 for x in output]

        if self.biasing:
            output.pop()
        return output


@np.vectorize
def sigmoid(x):
    return 1 / (1 + math.exp(-x * 1.0))

def ReLU(x):
    return [max(0.0, pred) for pred in x]
