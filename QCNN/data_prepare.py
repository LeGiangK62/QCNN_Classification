import numpy as np
import tensorflow as tf


def data_load(dataset, classes=[0, 1]):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data

    # Take only classes in classes
    train_mask = np.where((y_train == classes[0]) | (y_train == classes[1]))
    test_mask = np.where((y_test == classes[0]) | (y_test == classes[1]))

    X_train, X_test = x_train[train_mask], x_test[test_mask]
    Y_train, Y_test = y_train[train_mask], y_test[test_mask]

    Y_train = [1 if y == classes[0] else 0 for y in Y_train]
    Y_test = [1 if y == classes[0] else 0 for y in Y_test]

    X_train = tf.image.resize(X_train[:], (256, 1)).numpy()
    X_test = tf.image.resize(X_test[:], (256, 1)).numpy()
    X_train, X_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()

    return X_train, X_test, Y_train, Y_test

