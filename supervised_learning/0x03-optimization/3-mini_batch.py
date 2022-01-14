#!/usr/bin/env python3

"""
3-mini_batch module
"""
import numpy as np
import tensorflow.compat.v1 as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def print_stats(printing_d):
    """
    print_stats - prints the trainning stats.

    @printing_d: dictionary with all the trainning stats like:
        *epoch: <numeric> current epoch
        *train_cost: <numeric> cost of the model on the entire training set
        *train_accuracy: <numeric> the accuracy of the model on the entire
                            training set.
        *valid_cost: <numeric> the cost of the model on the entire
                     validation set.
        *valid_accuracy: <numeric> the accuracy of the model on the entire
                        validation set.

    Returns: None
    """

    print("After {epoch} epochs:".format(**printing_d))
    print("\tTraining Cost: {train_cost}".format(**printing_d))
    print("\tTraining Accuracy: {train_accuracy}".format(**printing_d))
    print("\tValidation Cost: {valid_cost}".format(**printing_d))
    print("\tValidation Accuracy: {valid_accuracy}".format(**printing_d))


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    train_mini_batch - trains a loaded neural network model using mini-batch
                       gradient descent.

    @X_train: a numpy.ndarray of shape (m, 784) containing the training data.
        *m: is the number of data points.
        *784: is the number of input features

    @Y_train: a one-hot numpy.ndarray of shape (m, 10) containing
              the training labels.
        *10: is the number of classes the model should classify

    @X_valid: numpy.ndarray of shape (m, 784) containing the validation data.

    @Y_valid: a one-hot numpy.ndarray of shape (m, 10) containing the
              validation labels.

    @batch_size: the number of data points in a batch.

    @epochs: the number of times the training should pass through
             the whole dataset.

    @load_path: the path from which to load the model.

    @save_path: the path to where the model should be saved after training.

    Returns: the path where the model was saved.
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        steps = list(range(0, X_train.shape[0], batch_size))
        if steps[-1] < X_train.shape[0]:
            steps.append(X_train.shape[0] + 1)
        batchs = list(zip(steps[:-1], steps[1:]))

        printing_d = {'epoch': 0,
                      'train_cost': 0,
                      'train_accuracy': 0,
                      'valid_cost': 0,
                      'valid_accuracy': 0}
        feed_dict = {x: X_train, y: Y_train}
        printing_d['train_accuracy'] = sess.run(accuracy, feed_dict)
        printing_d['train_cost'] = sess.run(loss, feed_dict)
        feed_dict = {x: X_valid, y: Y_valid}
        printing_d['valid_accuracy'] = sess.run(accuracy, feed_dict)
        printing_d['valid_cost'] = sess.run(loss, feed_dict)
        print_stats(printing_d)

        X_train, Y_train = shuffle_data(X_train, Y_train)
        for epoch in range(epochs):
            step = 1
            for start, end in batchs:
                x_batch = X_train[start:end]
                y_batch = Y_train[start:end]
                sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

                if step % 100 == 0:
                    print('span: {},{}'.format(start, end))
                    feed_dict = {x: x_batch, y: y_batch}
                    b_cost = sess.run(loss, feed_dict)
                    b_accuracy = sess.run(accuracy, feed_dict)
                    print('\tStep {}:'.format(step))
                    print('\t\tCost: {}'.format(b_cost))
                    print('\t\tAccuracy: {}'.format(b_accuracy))

                step += 1
            print('Last Batch: {},{}'.format(start, end))
            printing_d['epoch'] = epoch + 1
            feed_dict = {x: X_train, y: Y_train}
            printing_d['train_accuracy'] = sess.run(accuracy, feed_dict)
            printing_d['train_cost'] = sess.run(loss, feed_dict)
            feed_dict = {x: X_valid, y: Y_valid}
            printing_d['valid_accuracy'] = sess.run(accuracy, feed_dict)
            printing_d['valid_cost'] = sess.run(loss, feed_dict)
            print_stats(printing_d)

        return saver.save(sess, save_path)
