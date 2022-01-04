#!/usr/bin/env python3

"""
DeepNeuralNetwork module - defines a deep neural network performing
                        binary classification.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


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
        self.layers = layers

        # Dictionary to hold all intermediary values of the network.
        self.__cache = {}

        # Dictionary to hold all weights and biased of the network.
        self.__weights = {}
        for i in range(len(layers)):
            # position 0 of @layers correspond to the layer 1...
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])

            # offset the @i to correspond i=0 with W1.
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

    def cost(self, Y, A):
        """
        cost(Y, A) - Calculates the model cost using logistic regression.

        @Y: a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
        @A: a numpy.ndarray with shape (1, m) containing the activated
            output of the DeepNN for each example.

        Returns: the cost.
        """

        m = Y.shape[1]
        cost = - np.sum(
            np.multiply(Y, np.log(A)) +
            np.multiply((1 - Y), np.log(1.0000001 - A))
        ) / m

        return cost

    def evaluate(self, X, Y):
        """
        evaluate(X, Y) - Evaluates the neural network's predictions.

        @X: a numpy.ndarray with shape (nx, m) that contains the input data.
            *nx is the number of input features to the neuron.
            *m is the number of examples.
        @Y: a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.

        Returns: the neuron's prediction and the cost of the network,
                 respectively.
        """

        self.forward_prop(X)

        return np.where(self.cache['A' + str(self.L)] >= 0.5, 1, 0), \
            self.cost(Y, self.cache['A' + str(self.L)])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        gradient_descent(Y, cache, alpha=0.05) - Calculates one pass of
                                    gradient descent on the neural network.
        @Y: a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
        @cache: a dictionary containing all the intermediary values of
                the network.
        @alpha: is the learning rate.

        Returns: None. Updates the private attribute __weights.
        """

        m = Y.shape[1]

        dZl = cache['A' + str(self.L)] - Y
        for layer in range(self.L, 0, -1):
            Al_str = 'A' + str(layer - 1)
            Wl_str = 'W' + str(layer)
            bl_str = 'b' + str(layer)

            dWl = np.matmul(dZl, cache[Al_str].T) / m
            dbl = np.sum(dZl, axis=1, keepdims=True) / m
            dAl = cache[Al_str] * (1 - cache[Al_str])
            dZl = np.matmul(self.weights[Wl_str].T, dZl) * dAl

            self.__weights[Wl_str] -= alpha * dWl
            self.__weights[bl_str] -= alpha * dbl

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        train(X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100) - Trains the neural network.

        @X: a numpy.ndarray with shape (nx, m) that contains the input data.
            *nx is the number of input features to the neuron.
            *m is the number of examples.

        @Y: a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
            *m is the number of examples.

        @iterations: is the number of iterations to train over.

        @alpha: is the learning rate.

        @verbose: a boolean that defines whether or not to print information
                  about the training. If True, prints
                  "Cost after {iteration} iterations: {cost}"
                  every step iterations.
                  (Data from the 0th and last iteration is included)

        @graph: a boolean that defines whether or not to graph information
                about the training once the training has completed.
                If True:
                The training data is plotted every step iterations as
                a blue line. The x-axis is the iteration, the y-axis is
                the cost value. Data from 0th and last iteration is included.

        Returns: the evaluation of the training data after @iterations of
                 training have occurred.
                 The private attributes __weights and __cache are updated.
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        x_iteration = np.arange(0, iterations + step, step)
        y_costs = []

        for i in range(iterations + 1):
            self.forward_prop(X)

            if i in x_iteration:
                cost = self.cost(Y, self.cache['A' + str(self.L)])
                y_costs.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

            self.gradient_descent(Y, self.cache, alpha=alpha)

        if graph:
            plt.plot(x_iteration, y_costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        save(filename) - Saves the instance object to a file in pickle format

        @filename: is the file to which the object should be saved.
            *If filename does not have the extension .pkl, it is added.
        """

        if filename is None or filename == '':
            return None

        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=3)

    @staticmethod
    def load(filename):
        """
        load(filename) - Loads a pickled DeepNeuralNetwork object.

        @filename: the file from which the object should be loaded.

        Returns: the loaded object, or None if filename doesn't exist
        """

        if filename is None or filename == '':
            return None

        if not filename.endswith('.pkl'):
            return None

        if not os.path.isfile(filename):
            return None

        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f, fix_imports=True)
            return obj
        except FileNotFoundError:
            return None
