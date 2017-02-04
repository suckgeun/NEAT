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

    @staticmethod
    def is_connect_exist_nn(node_in, node_out, nn):
        """
        check if the connection between node_in and node_out exists

        :param node_in:
        :param node_out:
        :param nn: Neural network instance
        :return: True if exists, False if DNE
        """

        assert type(nn) == NeuralNetwork, "nn must be an instance of Neural Network"

        if nn.connect_genes is None:
            return False

        connect = [node_in, node_out]
        history = nn.connect_genes[:, :2]

        return any(np.equal(connect, history).all(1))

    def is_connect_exist_global(self, node_in, node_out):
        """
        check if connection (node_in, node_out) exists in all neural networks history.

        If connection exists, return its innovation number.
        If connection does not exits, return None

        :param node_in: index of input node
        :param node_out: index of output node
        :return: exists: innovation number, DNE: None
        """
        return self.workplace.innov_history.get((node_in, node_out))

    def is_input_node(self, node):
        """
        check if given node is input node.

        :param node: index of node
        :return: if input node, return true, if no, return false
        """
        assert node > -1, "node index must be positive integer"

        return node < self.workplace.n_input

    def is_output_node(self, node):
        """
        check if given node is output node

        :param node: index of node
        :return: if output node, return true, if no, return false
        """
        assert node > -1, "node index must be positive integer"

        is_input_node = self.is_input_node(node)
        n_input_output = self.workplace.n_input + self.workplace.n_output
        is_output_index = not is_input_node and node < n_input_output

        return is_output_index

    def is_in_in_connect(self, node1, node2):
        """
        check if node1 and node2 are both input nodes.
        :param node1:
        :param node2:
        :return: True if both are input nodes, False if one of them is not input node
        """

        assert node1 > -1 and node2 > -1,  "node index must be positive integer"

        is_node1_in = self.is_input_node(node1)
        is_node2_in = self.is_input_node(node2)

        return is_node1_in and is_node2_in

    def is_out_out_connect(self, node1, node2):
        """
        check if node1 and node2 are both output nodes.
        :param node1:
        :param node2:
        :return: True if both are output nodes, False if one of them is not output node
        """

        assert node1 > -1 and node2 > -1,  "node index must be positive integer"

        is_node1_out = self.is_output_node(node1)
        is_node2_out = self.is_output_node(node2)

        return is_node1_out and is_node2_out

    @staticmethod
    def is_recursive_connect(node_in, node_out):
        """
        check if node_in and node2 are the same; hence recursive connect.
        :param node_in:
        :param node_out:
        :return: True if recursive. False if not
        """

        assert node_in > -1 and node_out > -1, "node index must be positive integer"

        return node_in == node_out

    def add_connect(self, node_in, node_out, weight, enabled, innov_num, nn):

        assert node_in > -1 and node_out > -1, "node index must be positive integer"
        assert type(weight) is float, "weight must be float"
        assert enabled in (0, 1), "enabled must be 0 or 1"
        assert innov_num > -1, "innovation number must be positive integer"
        assert type(nn) is NeuralNetwork, "nn must be an instance of Neural Network"

        new_gene = np.array([[node_in, node_out, weight, enabled, innov_num]])

        if nn.connect_genes is None:
            nn.connect_genes = new_gene
        else:
            nn.connect_genes = np.vstack((nn.connect_genes, new_gene))

    def is_new_connect_valid(self, node_in, node_out, nn):

        assert node_in > -1, "node index must be positive integer"
        assert node_out > -1, "node index must be positive integer"
        assert type(nn) is NeuralNetwork, "nn must be instance of NeuralNetwork"

        # is in-in connection?
        self.is_input_node(node_in)
        # is out-out connection?
        # is recursive connection?
        # is it new to nn?
        # is it new to global?
        return False

    def create_connection_gene(self, node_in, node_out, weight, history):
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







