from players.neuralnet import NeuralNetwork
import numpy as np
from players.config import COL_IN, COL_OUT, COL_WEIGHT, COL_ENABLED, COL_INNOV, ENABLED
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

    def create_initial_info(self):
        """
        creates the initial gene, history, and counter using the number of inputs and outputs.

        :return: returns initial genes, history, counter
            assign those to workplace.
        """

        n_input = self.workplace.n_input
        n_output = self.workplace.n_output
        counter = -1
        history = {}
        genes = np.empty((n_input * n_output, 5), float)

        for node_in in range(n_input):
            for node_out in range(n_input, n_input + n_output):
                counter += 1

                genes[counter, COL_IN] = node_in
                genes[counter, COL_OUT] = node_out
                genes[counter, COL_WEIGHT] = random.uniform(-1.0, 1.0)
                genes[counter, COL_ENABLED] = ENABLED
                genes[counter, COL_INNOV] = counter

                history[(node_in, node_out)] = counter

        return genes, history, counter

    def initialize_workplace(self):
        """
        initialize nns, innov_history, and innov_counter in workplace

        :return:
        """

        # get init info
        gene, history, counter = self.create_initial_info()

        # init nns
        for _i in range(self.workplace.n_nn):
            nn = NeuralNetwork()
            nn.connect_genes = np.copy(gene)
            self.workplace.nns.append(nn)

        # init history
        self.workplace.innov_history = history

        # init counter
        self.workplace.innov_counter = counter

    def activate(self, xs, weights):
        """
        calculate the activation using the given xs, weights, and bias

        activation function workplace has will be used to calculate the output

        :param xs: input to neural networks
        :param weights: weights of each connection
        :return:
        """

        assert xs.ndim == 2 and xs.shape[0] == 1, "xs must be 2 dimensional array with one row"
        assert weights.ndim == 2 and weights.shape[1] == 1, "weights must be 2 dimensional array with one column"
        assert xs.shape[1] == weights.shape[0], "xs column len and weights row len must be the same"

        activ_func = self.workplace.activ_func
        bias = self.workplace.bias

        return activ_func(np.dot(xs, weights)) - bias





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










