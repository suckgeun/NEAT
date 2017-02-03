import numpy as np
import random
from players.config import COL_IN, COL_OUT, COL_WEIGHT, COL_ENABLED, COL_INNOV
from players.config import ENABLED, DISABLED, COL_TOTAL_NUM


class NeuralNetwork:
    """
    Neural Network class
    """

    def __init__(self):
        self.connect_genes = None

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








