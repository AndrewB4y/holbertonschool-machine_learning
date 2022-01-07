#!/usr/bin/env python3

"""
6-train module
"""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
         alpha, iterations, save_path="/tmp/model.ckpt")

    @X_train: a numpy.ndarray containing the training input data.
    @Y_train: a numpy.ndarray containing the training labels.
    @X_valid: a numpy.ndarray containing the validation input data.
    @Y_valid: a numpy.ndarray containing the validation labels.
    @layer_sizes: a list containing the number of nodes in each layer
                  of the network.
    @activations: a list containing the activation functions for each layer
                  of the network.
    @alpha: is the learning rate.
    @iterations: is the number of iterations to train over.
    @save_path: designates where to save the model.

    Returns: the path where the model was saved
    """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            t_accur = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            t_loss = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            v_accur = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            v_loss = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print('After {} iterations:'.format(i))
                print('\tTraining Cost: {}'.format(t_loss))
                print('\tTraining Accuracy: {}'.format(t_accur))
                print('\tValidation Cost: {}'.format(v_loss))
                print('\tValidation Accuracy: {}'.format(v_accur))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        return saver.save(sess, save_path)
