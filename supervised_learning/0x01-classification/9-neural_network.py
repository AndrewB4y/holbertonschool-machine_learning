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
