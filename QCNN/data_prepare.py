import numpy as np
import tensorflow as tf


def data_load(dataset, classes=[0, 1]):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data

    # Data Pre-processing
    X_train = x_train
    X_test = x_test

    # Take only 0 and 1 classes
    Y_train = [1 if y in classes else 0 for y in y_train]
    Y_test = [1 if y in classes else 0 for y in y_test]

    X_train = tf.image.resize(X_train[:], (256, 1)).numpy()
    X_test = tf.image.resize(X_test[:], (256, 1)).numpy()
    X_train, X_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()

    return X_train, X_test, Y_train, Y_test

