#!/usr/bin/env python3

"""
DeepNeuralNetwork module - defines a deep neural network performing
                        binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    DeepNeuralNetwork - defines a deep neural network performing
                        binary classification.
    """

    def __init__(self, nx, layers):
        """
        DeepNeuralNetwork(nx) - DeepNeuralNetwork constructor

        @nx: is the number of input features.
        @layers: a list representing the number of nodes in each layer of
                 the network.
                 The first value in layers represents the number of nodes
                 in the first layer, …
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Number of layers in the neural network.
        self.L = len(layers)

        # Dictionary to hold all intermediary values of the network.
        self.cache = {}

        # Dictionary to hold all weights and biased of the network.
        self.weights = {}
        for i in range(len(layers)):
            # position 0 of @layers correspond to the layer 1...
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                W = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                W = np.random.randn(
                    layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
            # offset the @i to correspond i=0 with W1.
            self.weights['W' + str(i + 1)] = W
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
