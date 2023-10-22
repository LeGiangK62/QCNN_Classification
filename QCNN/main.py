import numpy as np
import tensorflow as tf


# Hyperparameter
dataset = 'MNIST'
classes = [0, 1]
num_qubit = 8
encodder = 'AmplitudeEmbedding'
loss_fn = 'binaryCrossEntropy'

binary = 'True'

# Data loading
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Data Encoding


# Quantum Circuits


# Training


# Results


print("=== Done ===")