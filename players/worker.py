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

        # TODO rename to get_connect_innov_num

        return self.workplace.innov_history.get((node_in, node_out))

    def is_bias_node(self, node):
        """
        check if given node is bias

        :param node: node index
        :return:
        """
        assert node > -1, "node index must be positive integer"

        if self.workplace.bias is None:
            return False
        else:
            return node == 0

    def is_input_node(self, node):
        """
        check if given node is input node.

        :param node: index of node
        :return: if input node, return true, if no, return false
        """
        assert node > -1, "node index must be positive integer"

        is_bias = self.is_bias_node(node)

        n_input_bias = self.workplace.n_bias + self.workplace.n_input
        is_input_or_bias = node < n_input_bias

        return is_input_or_bias and not is_bias

    def is_output_node(self, node):
        """
        check if given node is output node

        :param node: index of node
        :return: if output node, return true, if no, return false
        """
        assert node > -1, "node index must be positive integer"

        is_bias = self.is_bias_node(node)
        is_input = self.is_input_node(node)
        n_total_node = self.workplace.n_input + self.workplace.n_output + self.workplace.n_bias
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
        assert type(nn) is NeuralNetwork, "nn must be an instance of Neural Network"
        assert not self.is_connect_exist_nn(node_in, node_out, nn), "connect must not exist in the neural network"

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

        # initialize node_index
        nn.node_index = list(range(n_node_in + n_node_out))

        # initialize connect_genes
        for node_in in range(n_node_in):
            # TODO change to get_output_nodes
            for node_out in range(n_node_in, n_node_in + n_node_out):

                # TODO decide how to cap the random weights.
                self.add_connect(node_in, node_out, random.uniform(-1.0, 1.0), nn)

        # initialize outputs_prev and outputs_cur
        outputs_prev = [None] * (n_input + n_output)
        outputs_cur = [None] * (n_input + n_output)
        if n_bias != 0:
            outputs_prev = [self.workplace.bias] + outputs_prev
            outputs_cur = [self.workplace.bias] + outputs_cur
        nn.outputs_prev = outputs_prev
        nn.outputs_cur = outputs_cur

    def initialize_workplace(self):
        """
        initialize workplace and its parameters including neural network list, node genes, history, and counter

        :return:
        """

        node_genes = self.workplace.node_genes_global
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

        self.workplace.is_initialized = True
        self.workplace.fitnesses = [None]*n_nn

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

    @staticmethod
    def get_weight_of_connect(node_in, node_out, nn):
        """
        get the weight of given connection.

        :param node_in:
        :param node_out:
        :param nn:
        :return: weight of given connection. if connection DNE, return None
        """

        connects = np.array(nn.connect_genes[:, 0:2])

        rows = np.where((connects == (node_in, node_out)).all(1))
        n_rows = len(rows[0])

        assert n_rows <= 1, "there are more then two identical connections. connection corrupted"

        if n_rows == 0:
            return None
        else:
            return nn.connect_genes[rows[0][0], 2]

    def calc_output(self, node_out, activ_result, inputs, nn):
        """
        calculate the output of the given node.

        it uses the input info of workplace

        :param node_out: node to calculate output
        :param activ_result: record for all result of activation of nodes.
        :param inputs:
        :param nn:
        :return: updated activ_result
        """

        # TODO: performance bottle neck. Hard to understand code

        assert not self.is_bias_node(node_out), "node_out cannot be bias node"
        assert not self.is_input_node(node_out), "node_out cannot be input node"

        n_bias = self.workplace.n_bias
        sum_wx = 0
        nodes_in = self.get_nodes_in_of_node(node_out, nn)
        for node in nodes_in:
            node = int(node)
            weight = self.get_weight_of_connect(node, node_out, nn)
            if activ_result[node] is None:
                if self.is_input_node(node):
                    activ_result[node] = inputs[node - n_bias]
                elif self.is_bias_node(node):
                    activ_result[node] = self.workplace.bias
                else:
                    self.calc_output(node, activ_result, inputs, nn)

            sum_wx += activ_result[node] * weight

        result = self.workplace.activ_func(sum_wx)
        activ_result[node_out] = result

        return activ_result

    def get_output_nodes(self):
        """
        get the output node index

        :return: list of output node index
        """
        n_input = self.workplace.n_input
        n_output = self.workplace.n_output
        n_bias = self.workplace.n_bias
        n_total = n_bias + n_input + n_output

        return [node for node in range(n_bias+n_input, n_total)]

    def feedforward(self, inputs, nn):
        """
        calculate all outputs of the neural network.

        :param inputs:
        :param nn:
        :return: outputs as list
        """

        n_input = self.workplace.n_input
        n_output = self.workplace.n_output
        n_bias = self.workplace.n_bias
        n_total = len(self.workplace.node_genes_global)
        activ_result = [None] * n_total

        outputs = self.get_output_nodes()
        for output in outputs:
            self.calc_output(output, activ_result, inputs, nn)

        return activ_result[(n_bias+n_input):(n_bias + n_input + n_output)]

    def get_node_between(self, node_in, node_out):
        """
        get one node between node_in and node_out.

        if there are multiple nodes between the given node_in and out, it returns whatever it could find first.
        :param node_in:
        :param node_out:
        :return: whatever it can find first. If there is no node between in and out, it returns None
        """

        # TODO: there must be a better way..

        outs = []
        for connect_in, connect_out in self.workplace.innov_history:
            if connect_in == node_in:
                outs.append(connect_out)

        for connect_in, connect_out in self.workplace.innov_history:
            if connect_in in outs and connect_out == node_out:
                return connect_in

        return None

    def add_node(self, node_in, node_out, nn):
        """
        add node in the given node_in, node_out connection of the neural network.

        :param node_in:
        :param node_out:
        :param nn:
        :return:
        """

        assert self.is_connect_exist_nn(node_in, node_out, nn), "connection must exist to add node in"
        assert nn.connect_genes is not None, "neural network must be initialized first"
        assert self.workplace.is_initialized, "workplace must be initialized first"

        ori_weight = self.get_weight_of_connect(node_in, node_out, nn)

        new_node = self.get_node_between(node_in, node_out)
        if new_node is None:
            self.workplace.node_genes_global.append(3)
            new_node = len(self.workplace.node_genes_global) - 1

        self.add_connect(node_in, new_node, 1.0, nn)
        self.add_connect(new_node, node_out, ori_weight, nn)

    def disable_connect(self, node_in, node_out, nn):
        """
        disable connect

        :param node_in:
        :param node_out:
        :param nn:
        :return:
        """

        assert self.is_connect_exist_nn(node_in, node_out, nn), "connect must exist"

        for gene in nn.connect_genes:
            if gene[0] == node_in and gene[1] == node_out:
                gene[3] = 0

    def enable_connect(self, node_in, node_out, nn):
        """
        enable connect

        :param node_in:
        :param node_out:
        :param nn:
        :return:
        """

        assert self.is_connect_exist_nn(node_in, node_out, nn), "connect must exist"

        for gene in nn.connect_genes:
            if gene[0] == node_in and gene[1] == node_out:
                gene[3] = 1

    @staticmethod
    def get_matching_innov_num(nn1, nn2):
        """
        get a list of innovation numbers common in nn1 and nn2

        :param nn1:
        :param nn2:
        :return: list of matching innovation numbers. [] if none
        """
        match = []
        genes1 = nn1.connect_genes
        genes2 = nn2.connect_genes

        if genes1.shape[0] >= genes2.shape[0]:
            matching_size = genes2.shape[0]
        else:
            matching_size = genes1.shape[0]

        for i in range(matching_size):
            if genes1[i, 4] == genes2[i, 4]:
                match.append(genes1[i, 4])

        return match

    @staticmethod
    def inherit_match(match, nn1, nn2):
        """
        inherit matching genes randomly from nn1 and nn2.

        :param match: list of common innovation numbers in nn1 and nn2
        :param nn1:
        :param nn2:
        :return: matching connection genes randomly chosen from nn1 or nn2
        """

        genes1 = nn1.connect_genes
        genes2 = nn2.connect_genes
        genes_new = None

        for innov_num in match:
            rand = random.random()
            if rand > 0.5:
                gene_to_add = genes1[genes1[:, 4] == innov_num]
            else:
                gene_to_add = genes2[genes2[:, 4] == innov_num]

            if genes_new is None:
                genes_new = gene_to_add
            else:
                genes_new = np.vstack((genes_new, gene_to_add))

        return genes_new

    @staticmethod
    def inherit_disjoint_excess(match, nn1, nn2):
        """
        inherit disjoint and excess genes from more fit nn.

        if nn1 and nn2 have the same fitness, all disjoints and excesses from both nns are inherited

        :param match: matching genes of the two nns
        :param nn1:
        :param nn2:
        :return: inherited genes from nn1 and nn2
        """

        genes1 = nn1.connect_genes
        genes2 = nn2.connect_genes

        if nn1.fitness > nn2.fitness:
            genes_more_fit = genes1
        elif nn1.fitness < nn2.fitness:
            genes_more_fit = genes2
        else:
            genes_more_fit = None

        if genes_more_fit is None:
            return np.vstack((genes1[len(match):, :], genes2[len(match):, :]))
        else:
            return genes_more_fit[len(match):, :]

    def crossover(self, nn1, nn2):
        """
        crossover the two nns

        it creates a new neural network from nn1 and nn2.
        first, matching genes are inherited,
        then matching genes disability are modified using enablilty_preset,
        then disjoints and excess genes are inherited.

        :param nn1:
        :param nn2:
        :return: newly inherited neural network.
        """
        nn_new = NeuralNetwork()

        match = self.get_matching_innov_num(nn1, nn2)

        genes_matching = self.inherit_match(match, nn1, nn2)
        genes_disj_exc = self.inherit_disjoint_excess(match, nn1, nn2)
        genes_new = np.vstack((genes_matching, genes_disj_exc))

        nn_new.connect_genes = genes_new

        return nn_new

    def activate_neurons(self, inputs, nn):



        pass




















