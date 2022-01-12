#!/usr/bin/env python3

"""
1-create_layer module
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    create_layer(prev, n, activation) - Creates a layer with the given params

    @prev: is the tensor output of the previous layer.
    @n: is the number of nodes in the layer to create.
    @activation: is the activation function that the layer should use.

    Returns: the tensor output of the layer.
    """

    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(
        units=n, activation=activation, name='layer',
        kernel_initializer=init)

    return layer(prev)