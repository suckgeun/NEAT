from players.neuralnet import NeuralNetwork
import numpy as np
from players.config import ENABLED
import random


class Worker:

    def __init__(self, workplace):
        self.workplace = workplace

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

    @staticmethod
    def is_bias_node(node):
        """
        check if given node is bias

        :param node: node index
        :return:
        """
        assert node > -1, "node index must be positive integer"
        return node == 0

    def is_input_node(self, node):
        """
        check if given node is input node.

        :param node: index of node
        :return: if input node, return true, if no, return false
        """
        assert node > -1, "node index must be positive integer"

        is_bias = self.is_bias_node(node)
        is_input = node <= self.workplace.n_input

        return is_input and not is_bias

    def is_output_node(self, node):
        """
        check if given node is output node

        :param node: index of node
        :return: if output node, return true, if no, return false
        """
        assert node > -1, "node index must be positive integer"

        is_bias = self.is_bias_node(node)
        is_input = self.is_input_node(node)
        n_total_node = self.workplace.n_input + self.workplace.n_output + 1
        is_output = not is_bias and not is_input and node < n_total_node

        return is_output

    def is_bias_in_connect(self, node1, node2):
        """
        check if node_in and node_out are bias to in connect

        :param node1:
        :param node2:
        :return:
        """
        assert node1 > -1 and node2 > -1,  "node index must be positive integer"

        is_bias1 = self.is_bias_node(node1)
        is_bias2 = self.is_bias_node(node2)
        is_iniput1 = self.is_input_node(node1)
        is_iniput2 = self.is_input_node(node2)

        return (is_bias1 and is_iniput2) or (is_bias2 and is_iniput1)

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

    def is_in_bias_at_end_connect(self, node_in, node_out):
        """
        check if end of connection is bias or input node

        :param node_in:
        :param node_out:
        :return:
        """
        assert node_in > -1 and node_out > -1,  "node index must be positive integer"

        is_bias = self.is_bias_node(node_out)
        is_input = self.is_input_node(node_out)

        return is_bias or is_input

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

    def increment_innov_counter(self):
        """
        increment the global innovation counter

        :return:
        """
        self.workplace.innov_counter += 1

    def record_innov_history(self, connect):
        """
        record the connection to the innov_history with the current innov_counter

        :param connect: (node_in, node_out) tuple
        :return:
        """
        assert connect not in self.workplace.innov_history, "the connection already exists in the innovation history"
        assert type(connect) is tuple, "connect must be tuple (node_in, node_out)"

        self.workplace.innov_history[connect] = self.workplace.innov_counter

    def add_connect(self, node_in, node_out, weight, nn):
        """
        add connection to the given neural network.

        if connection already exists in innov_history, the innov_number will be assigned to the new gene.
        if connection does not exist in history, innov_counter will be incremented, and history will be recorded

        :param node_in:
        :param node_out:
        :param weight:
        :param nn:
        :return:
        """

        assert node_in > -1 and node_out > -1, "node index must be positive integer"
        assert type(weight) is float, "weight must be float"
        assert type(nn) is NeuralNetwork, "nn must be an instance of Neural Network"
        assert not self.is_connect_exist_nn(node_in, node_out, nn), "connect must not exist in the neural network"
        assert not self.is_in_in_connect(node_in, node_out), "both nodes cannot be input nodes"
        assert not self.is_out_out_connect(node_in, node_out), "both nodes cannot be output nodes"
        assert not self.is_recursive_connect(node_in, node_out), "recursive connect not allowed"
        assert not self.is_bias_in_connect(node_in, node_out), "both node cannot be bias and input node"
        assert not self.is_in_bias_at_end_connect(node_out, node_out), "output node cannot be bias or input node"

        innov_num = self.is_connect_exist_global(node_in, node_out)

        if innov_num is None:
            self.increment_innov_counter()
            self.record_innov_history((node_in, node_out))
            innov_num = self.workplace.innov_counter

        new_gene = np.array([[node_in, node_out, weight, ENABLED, innov_num]])

        if nn.connect_genes is None:
            nn.connect_genes = new_gene
        else:
            nn.connect_genes = np.vstack((nn.connect_genes, new_gene))

    def initialize_nn(self, nn):
        """
        initialize the given neural network.

        :param nn:
        :return:
        """

        n_input = self.workplace.n_input
        n_output = self.workplace.n_output
        n_bias = self.workplace.n_bias

        n_node_in = n_bias + n_input
        n_node_out = n_output

        for node_in in range(n_node_in):
            for node_out in range(n_node_in, n_node_in + n_node_out):

                # TODO decide how to cap the random weights.
                self.add_connect(node_in, node_out, random.uniform(-1.0, 1.0), nn)

    def initialize_workplace(self):
        """
        initialize workplace and its parameters including neural network list, node genes, history, and counter

        :return:
        """

        node_genes = self.workplace.node_genes
        n_bias = self.workplace.n_bias
        n_input = self.workplace.n_input
        n_output = self.workplace.n_output
        n_nn = self.workplace.n_nn

        for _i in range(n_bias):
            node_genes.append(0)

        for _i in range(n_input):
            node_genes.append(1)

        for _i in range(n_output):
            node_genes.append(2)

        for _i in range(n_nn):
            nn = NeuralNetwork()
            self.initialize_nn(nn)
            self.workplace.nns.append(nn)

    def activate(self, xs, weights):
        """
        calculate the activation using the given xs and weights

        activation function workplace has will be used to calculate the output

        :param xs: input to neural networks
        :param weights: weights of each connection
        :return:
        """

        assert xs.ndim == 2 and xs.shape[0] == 1, "xs must be 2 dimensional array with one row"
        assert weights.ndim == 2 and weights.shape[1] == 1, "weights must be 2 dimensional array with one column"
        assert xs.shape[1] == weights.shape[0], "xs column len and weights row len must be the same"

        activ_func = self.workplace.activ_func

        return activ_func(np.dot(xs, weights))

    @staticmethod
    def get_nodes_in_of_node(node_out, nn):
        """
        gets all the node_in of given node using the connect_genes of neural network

        :param node_out:
        :param nn:
        :return:
        """

        connects = nn.connect_genes[:, 0:2]

        return {connect[0] for connect in connects if connect[1] == node_out}

    # def calc_output(self, node_out, inputs, nn):
    #
    #     nodes_in = self.get_nodes_in_of_node(node_out, nn)
    #
    #     for node_in in nodes_in:
    #         is_input_node = self.is_input_node(node_in)
    #         is_bias = self.is_bias_node(node_in)
    #         is_
    #
    #
    #     pass

    def feedforward(self, input_data, nn):
        sum_wx = 0

        # get all inputs of node_out
        # for input in inputs
        ## if input result is not none
        ### sum_wx += result * weight
        ## else
        ### feedforward(input, nn)

        # return self.activate(sum_wx)
        pass












