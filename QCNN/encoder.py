# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# Source: http://github.com/takh04/QCNN
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding


def data_embedding(x, embedding_type='Amplitude'):
    if embedding_type == 'Amplitude':
        AmplitudeEmbedding(x, wires=range(8), normalize=True)
    elif embedding_type == 'Angle':
        AngleEmbedding(x, wires=range(8), rotation='Y')
    elif embedding_type == 'Angle-compact':
        AngleEmbedding(x[:8], wires=range(8), rotation='X')
        AngleEmbedding(x[8:16], wires=range(8), rotation='Y')

