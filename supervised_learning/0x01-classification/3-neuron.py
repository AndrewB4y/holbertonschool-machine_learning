#!/usr/bin/env python3

"""
Neuron module - module to define a Neuron class that behaves as an AI neuron
"""

import numpy as np


class Neuron:
    """
    Neuron class - defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Neuron(nx) - Neuron class constructor

        @nx: is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        W - property getter for private attribute W
        """
        return self.__W

    @property
    def b(self):
        """
        b - property getter for private attribute b
        """
        return self.__b

    @property
    def A(self):
        """
        A - property getter for private attribute A
        """
        return self.__A

    def forward_prop(self, X):
        """
        forward_prop(X) - Calculates the forward propagation of the neuron.

        @X: is a numpy.ndarray with shape (nx, m), contains the input data.
                nx is the number of input features to the neuron
                m is the number of examples

        Returns: the private attribute __A
        """

        out = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-out))

        return self.__A

    def cost(self, Y, A):
        """
        cost(Y, A) - Calculates the cost of the model using
                     logistic regression.
        @Y: is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
        @A: is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example.

        Returns: the cost
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
