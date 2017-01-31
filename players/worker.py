from players.neuralnet import NeuralNetwork
import numpy as np
from players.config import COL_IN, COL_OUT, COL_WEIGHT, COL_ENABLED, COL_INNOV, COL_TOTAL_NUM, DISABLED, ENABLED
import random


class Worker:

    def __init__(self, workplace):
        self.workplace = workplace

    def initialize_nn(self, n_input, n_output):
        nn = NeuralNetwork()
        nn.node_genes = np.array([0] * n_input + [1] * n_output)
        nn.connect_genes = np.empty((n_input * n_output, 5), float)
        new_counter = 0
        new_history = np.array([])

    def connect_gene_exits(self, node_in, node_out, history):
        """
        check if connection between node_in and node_out exists in all neural networks history.

        If connection exists, return its global innovation counter.
        If connection does not exits, return -1

        :param node_in: index of input node
        :param node_out: index of output node
        :param history: record of connections of all. 2D array with each row represents one history,
                and row is [in, out, counter]
                ex) [[1, 2, 3], [2, 4, 5], ...]
        :return: exists: global counter, DNE: -1
        """
        assert history.shape[1] == 3, "history must be numpy array and have 3 columns"

        connection = [node_in, node_out]

        element_compared = np.equal(connection, history[:, :2])
        row_compared = np.all(element_compared, 1)
        matching_row = np.where(row_compared)[0]

        assert matching_row.shape[0] < 2, "history corrupted. more then one same gene in history"

        if matching_row.shape[0] == 1:
            index = matching_row[0]
            return history[index][2]
        else:
            return -1

    def create_connection_gene(self, node_in, node_out, weight):
        pass



    # @staticmethod
    # def initialize_nn(n_input, n_output, counter, history):
    #     """
    #     initialize one Neural Network with the given info
    #
    #     :param n_input: number of inputs
    #     :param n_output: number of outputs
    #     :param counter: global innovation counter
    #     :param history: global innovation history
    #     :return: initialized neural network, incremented counter, and incremented history
    #     """
    #     nn = NeuralNetwork()
    #     nn.node_genes = np.array([0] * n_input + [1] * n_output)
    #     nn.connect_genes = np.empty((n_input * n_output, 5), float)
    #     new_counter = 0
    #     new_history = np.array([])
    #
    #
    #     row = 0
    #     for innode in range(n_input):
    #         for outnode in range(n_output):
    #             nn.connect_genes[row, COL_IN] = innode
    #             nn.connect_genes[row, COL_OUT] = outnode + n_input
    #             nn.connect_genes[row, COL_WEIGHT] = random.uniform(-1.0, 1.0)
    #             nn.connect_genes[row, COL_ENABLED] = ENABLED
    #             nn.connect_genes[row, COL_INNOV] = new_counter
    #             row += 1
    #             new_counter += 1
    #
    #     return nn, new_counter, new_history


