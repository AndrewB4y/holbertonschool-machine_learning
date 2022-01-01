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
        self.__L = len(layers)

        # Dictionary to hold all intermediary values of the network.
        self.__cache = {}

        # Dictionary to hold all weights and biased of the network.
        self.__weights = {}
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
            self.__weights['W' + str(i + 1)] = W
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        L - property getter for private attribute L
        """
        return self.__L

    @property
    def cache(self):
        """
        cache - property getter for private attribute cache
        """
        return self.__cache

    @property
    def weights(self):
        """
        weights - property getter for private attribute weights
        """
        return self.__weights

    def forward_prop(self, X):
        """
        forward_prop(self, X) - Calculates the forward propagation of the
                                neural network (NN).
                                Updates the private attribute __cache.
        @X: a numpy.ndarray with shape (nx, m) that contains the input data.
            *nx is the number of input features to the neuron.
            *m is the number of examples.

        Returns: the output of the NN and the cache, respectively.
        """

        self.__cache['A0'] = X
        for layer in range(1, self.L + 1):
            Wl = self.weights['W' + str(layer)]
            bl = self.weights['b' + str(layer)]
            layer_Z = np.matmul(Wl, self.cache['A' + str(layer - 1)]) + bl
            self.__cache['A' + str(layer)] = 1 / (1 + np.exp(-layer_Z))

        return self.cache['A' + str(layer)], self.cache
