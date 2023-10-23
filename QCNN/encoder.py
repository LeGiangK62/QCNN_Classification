# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# Source: http://github.com/takh04/QCNN
from pennylane.templates.embeddings import AmplitudeEmbedding


def data_embedding(x, num_qubits=8):
    AmplitudeEmbedding(x, wires=range(num_qubits), normalize=True)

