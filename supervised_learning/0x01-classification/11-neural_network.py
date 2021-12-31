#!/usr/bin/env python3

"""
NeuralNetwork module - defines a neural network with one hidden layer
                       performing binary classification.
"""

import numpy as np


class NeuralNetwork:
    """
    NeuralNetwork - defines a neural network with one hidden layer
                    performing binary classification.
    """

    def __init__(self, nx, nodes):
        """
        NeuralNetwork(nx) - NeuralNetwork constructor

        @nx: is the number of input features.
        @nodes: is the number of nodes found in the hidden layer.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        W1 - property getter for private attribute W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        b1 - property getter for private attribute b1
        """
        return self.__b1

    @property
    def A1(self):
        """
        A1 - property getter for private attribute A1
        """
        return self.__A1

    @property
    def W2(self):
        """
        W2 - property getter for private attribute W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        b2 - property getter for private attribute b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        A2 - property getter for private attribute A2
        """
        return self.__A2

    def forward_prop(self, X):
        """
        forward_prop(self, X) - Calculates the forward propagation of the
                                neural network.
        @X: a numpy.ndarray with shape (nx, m) that contains the input data.
            *nx is the number of input features to the neuron.
            *m is the number of examples.

        Returns: the private attributes __A1 and __A2, respectively.
        """

        out1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-out1))

        out2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-out2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        cost(Y, A) - Calculates the model cost using logistic regression.

        @Y: a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
        @A: a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example.

        Returns: the cost.
        """

        m = Y.shape[1]
        cost = - np.sum(
            Y * np.log(A) + ((1 - Y) * np.log(1.0000001 - A))
        ) / m

        """
        Squeeze to make sure your cost's shape is what we expect:
        (e.g. this turns [[17]] into 17).
        """
        cost = np.squeeze(cost)

        return cost
