import numpy as np
import random
from players.config import COL_IN, COL_OUT, COL_WEIGHT, COL_ENABLED, COL_INNOV
from players.config import ENABLED, DISABLED, COL_TOTAL_NUM


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
        assert n_input > 0, "invalid n_input"
        assert n_output > 0, "invalid n_output"

        self.node_genes = np.array([0] * n_input + [1] * n_output)
        self.connect_genes = np.empty((n_input * n_output, 5), float)

        row = 0
        for innode in range(n_input):
            for outnode in range(n_output):
                self.connect_genes[row, COL_IN] = innode
                self.connect_genes[row, COL_OUT] = outnode + n_input
                self.connect_genes[row, COL_WEIGHT] = random.uniform(-1.0, 1.0)
                self.connect_genes[row, COL_ENABLED] = ENABLED
                self.connect_genes[row, COL_INNOV] = row
                row += 1

    def add_hidden_node(self, target_connection=None):

        # test if target connection is valid
        n_connections = self.connect_genes.shape[0]
        assert target_connection > -1, "target_connection must be positive"
        assert target_connection < n_connections, "target_connection exceeds the number of connections"

        # if target_connection is None, choose a random connection
        if target_connection is None:
            target_connection = random.randint(0, n_connections-1)

        # increase node_gene
        self.node_genes = np.append(self.node_genes, [2])

        # disable the old connection
        self.connect_genes[target_connection, COL_ENABLED] = DISABLED

        old_innode = self.connect_genes[target_connection, COL_IN]
        old_outnode = self.connect_genes[target_connection, COL_OUT]
        old_weight = self.connect_genes[target_connection, COL_WEIGHT]
        new_node_index = len(self.node_genes) - 1

        # insert new connections
        # first, create connection between input and new hidden
        connection_front = np.empty(COL_TOTAL_NUM, float)
        connection_front[COL_IN] = old_innode
        connection_front[COL_OUT] = new_node_index
        connection_front[COL_WEIGHT] = 1.0
        connection_front[COL_ENABLED] = 1
        # TODO increment global innovation number
        connection_front[COL_INNOV] = 0

        # create connection between new hidden and output
        connection_end = np.empty(COL_TOTAL_NUM, float)
        connection_end[COL_IN] = new_node_index
        connection_end[COL_OUT] = old_outnode
        connection_end[COL_WEIGHT] = old_weight
        connection_end[COL_ENABLED] = 1
        # TODO increment global innovation number
        connection_end[COL_INNOV] = 0

        # add two rows to the connection gene
        self.connect_genes = \
            np.vstack((self.connect_genes, connection_front, connection_end))








