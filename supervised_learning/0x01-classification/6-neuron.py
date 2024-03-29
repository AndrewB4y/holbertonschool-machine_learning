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
            *m: is the number of examples
        @A: is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example.
            *m: is the number of examples

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

    def evaluate(self, X, Y):
        """
        evaluate(X, Y) - Evaluates the neuron's predictions.

        @X: is a numpy.ndarray with shape (nx, m) that contains
            the input data.
            *nx: is the number of input features to the neuron
            *m: is the number of examples
        @Y: is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.

        Returns: the neuron's prediction and the cost of the network,
                 respectively.
                 The prediction should be a numpy.ndarray with shape (1, m)
                 containing the predicted labels for each example.
                 The label values should be 1 if the output of the network is
                 >= 0.5 and 0 otherwise.
        """

        prediction = self.forward_prop(X)

        cost = self.cost(Y, prediction)

        prediction = np.where(prediction >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        gradient_descent(X, Y, A, alpha=0.05) - Calculates one pass of
                                gradient descent on the neuron.
                                Updates the private attributes __W and __b.

        @X: a numpy.ndarray with shape (nx, m) that contains the input data.
            *nx is the number of input features to the neuron.
            *m is the number of examples.
        @Y: a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
        @A: a numpy.ndarray with shape (1, m) containing the activated output
            of the neuron for each example.
        @alpha: is the learning rate.

        Return: None
        """

        m = Y.shape[1]

        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        train(X, Y, iterations=5000, alpha=0.05) - Trains the neuron.

        @X: a numpy.ndarray with shape (nx, m) that contains the input data.
            *nx is the number of input features to the neuron.
            *m is the number of examples.

        @Y: a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
            *m is the number of examples.

        @iterations: is the number of iterations to train over.
        @alpha: is the learning rate.

        Returns: the evaluation of the training data after iterations of
                 training have occurred.
                 The private attributes __W, __b, and __A are updated.
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha=alpha)

        n_eval = self.evaluate(X, Y)

        return n_eval
