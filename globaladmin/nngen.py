from players.neuralnet import NeuralNetwork
import numpy as np

INNOVATION_COUNTER = 0


def create_nns(n_input, n_output, n_nn):
    """
    creates a list of total neural networks.

    input nodes are described as 0
    output nodes are describes as 1

    :param n_input: number of input nodes
    :param n_output: number of output nodes
    :param n_nn: number of total neural networks
    :return: list of created neural networks
    """
    nns = []
    for i in range(n_nn):
        nn = NeuralNetwork()
        nn.node_genes = np.array([0]*n_input + [1]*n_output)
        nn.connect_genes = np.array([])
        nns.append(nn)

    return nns