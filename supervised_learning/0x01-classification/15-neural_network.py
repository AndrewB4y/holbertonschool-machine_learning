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

        A1, A2 = self.forward_prop(X)

        cost = self.cost(Y, A2)

        prediction = np.where(A2 >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        gradient_descent(X, Y, A1, A2, alpha=0.05) - Calculates one pass of
                                    gradient descent on the neural network
        @X: a numpy.ndarray with shape (nx, m) that contains the input data.
            *nx is the number of input features to the neuron.
            *m is the number of examples.
        @Y: a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
        @A1: is the output of the hidden layer.
        @A2: is the predicted output.
        @alpha: is the learning rate.

        Returns: None. Updates the private attributes
                 __W1, __b1, __W2, and __b2.
        """

        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = A1 * (1 - A1)
        dZ1 = np.matmul(self.__W2.T, dZ2) * dA1
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

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
                 The private attributes __W1, __b1, __A1, __W2, __b2,
                 and __A2 are updated.
        """
        import matplotlib.pyplot as plt

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

        x_iteration = []
        y_costs = []
        if verbose:
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)
            print("Cost after {} iterations: {}".format(0, cost))
            if graph:
                x_iteration.append(0)
                y_costs.append(cost)

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha=alpha)

            if verbose:
                if i != 0 and i % step == 0:
                    cost = self.cost(Y, A2)
                    print("Cost after {} iterations: {}".format(i, cost))
                    if graph:
                        x_iteration.append(i)
                        y_costs.append(cost)

        if verbose:
            cost = self.cost(Y, A2)
            print("Cost after {} iterations: {}".format(i + 1, cost))
            if graph:
                x_iteration.append(i)
                y_costs.append(cost)
                plt.plot(x_iteration, y_costs, 'b-')
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.title("Training Cost")
                plt.show()

        n_eval = self.evaluate(X, Y)

        return n_eval
