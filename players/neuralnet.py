import numpy as np
import random


class NeuralNetwork:
    """
    Neural Network class

    in node_gene, input nodes are expressed as 0, output nodes are 1, and hidden nodes are 2.
    """

    def __init__(self, n_input, n_output):
        """
        initialize neural network.

        example) n_input = 2 and n_output = 3,
        * node_gene:  [0, 0, 1, 1, 1]
        * connect_gene:
        # in, out, weight, enabled, innov
        #  0,   2,     w1,       1,     0,
        #  0,   3,     w2,       1,     1,
        #  0,   4,     w3,       1,     2,
        #  1,   2,     w4,       1,     3,
        #  1,   3,     w5,       1,     4,
        #  1,   4,     w6,       1,     5,

        where weights are random float numbers (-1, 1)

        :param n_input: number of input nodes
        :param n_output: number of output nodes
        """

        self.node_genes = np.array([0] * n_input + [1] * n_output)
        self.connect_genes = np.empty((n_input * n_output, 5), float)

        row = 0
        for innode in range(n_input):
            for outnode in range(n_output):
                self.connect_genes[row, 0] = innode
                self.connect_genes[row, 1] = outnode + n_input
                self.connect_genes[row, 2] = random.uniform(-1.0, 1.0)
                self.connect_genes[row, 3] = 1
                self.connect_genes[row, 4] = row
                row += 1

