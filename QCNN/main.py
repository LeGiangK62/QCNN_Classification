import quantum_circuit
import data_prepare
from pennylane import numpy as np
import tensorflow as tf
import pennylane as qml
import autograd.numpy as anp
from pennylane.templates.embeddings import AmplitudeEmbedding


# Hyperparameter
dataset = 'MNIST'
classes = [0, 1]
num_qubits = 8
encodder = 'AmplitudeEmbedding'
loss_fn = 'binaryCrossEntropy'


# Cost function
def cross_entropy(labels, preds):
    loss = 0
    for l, p in zip(labels, preds):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss


def cost(parameters, X, Y, U, U_parameters, num_qubits=8):
    preds = [quantum_circuit.QCNN(x, parameters, U, U_parameters, num_qubits) for x in X]
    loss = cross_entropy(Y, preds)

    return loss


def accuracy_test(preds, labels):
    acc = 0
    for l,p in zip(labels, preds):
        if p[0] > p[1]:
            P = 0
        else:
            P = 1
        if P == l:
            acc = acc + 1
    return acc / len(labels)


# Data loading
X_train, X_test, Y_train, Y_test = data_prepare.data_load(dataset, classes=classes)


# Training
steps = 100
learning_rate = 0.01
batch_size = 25


U = 'U_SU4'
U_params = 15
total_params = U_params * 3 + 2 * 3


params = np.random.randn(total_params, requires_grad=True)
opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
loss_history = []

for it in range(steps):
    batch_index = np.random.randint(0, len(X_train), (batch_size,))
    X_batch = [X_train[i] for i in batch_index]
    Y_batch = [Y_train[i] for i in batch_index]
    params, cost_new = opt.step_and_cost(
        lambda v: cost(v, X_batch, Y_batch, U, U_params, num_qubits),
        params)
    loss_history.append(cost_new)
    if it % 10 == 0:
        print("iteration: ", it, " cost: ", cost_new)
print("=== Done Training ===")
# Testing
trained_params = params
predictions = [quantum_circuit.QCNN(x, params, U, U_params, num_qubits) for x in X_test]

accuracy = accuracy_test(predictions, Y_test)
print("Accuracy for " + U + " :" + str(accuracy))

