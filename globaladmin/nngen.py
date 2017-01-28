from players.neuralnet import NeuralNetwork

INNOVATION_COUNTER = 0


def create_nns(n_input, n_output, n_nn):
    """
    creates a list of total neural networks.

    :param n_input: number of input nodes
    :param n_output: number of output nodes
    :param n_nn: number of total neural networks
    :return: list of created neural networks
    """
    nns = []
    for i in range(n_nn):
        nn = NeuralNetwork(n_input, n_output)
        nns.append(nn)
    global INNOVATION_COUNTER
    INNOVATION_COUNTER = n_input * n_output - 1

    return nns
