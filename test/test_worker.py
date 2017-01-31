import unittest
import numpy as np
from players.neuralnet import NeuralNetwork
from players.worker import Worker
from globaladmin.workplace import Workplace


class WorkerTest(unittest.TestCase):

    def test_connect_gene_exists__no_history(self):
        workplace = Workplace(n_input=3, n_output=1)
        worker = Worker(workplace)

        node_in = 1
        node_out = 3
        history = np.empty((1, 3))

        self.assertEqual(worker.connect_gene_exits(node_in, node_out, history), -1)

    def test_connect_gene_exists__yes_history(self):
        workplace = Workplace(n_input=3, n_output=1)
        worker = Worker(workplace)

        node_in = 2
        node_out = 3
        history = np.array([[1, 3, 1], [2, 3, 4]])

        self.assertEqual(worker.connect_gene_exits(node_in, node_out, history), 4)

    def test_add_connect_gene__no_history(self):
        workplace = Workplace(3, 1)
        worker = Worker(workplace)
        node_in = 1
        node_out = 3
        weight = 0.5
        workplace.innov_history = []

        gene = worker.create_connection_gene(node_in, node_out, weight)

        self.assertTrue(np.array_equal(gene, np.array([node_in, node_out, weight, 1, 0])))

    def test_add_connect_gene__yes_history(self):
        workplace = Workplace(3, 1)
        worker = Worker(workplace)
        node_in = 1
        node_out = 3
        weight = 0.5
        workplace.innov_history = [[1, 3]]

        gene = worker.create_connection_gene(node_in, node_out, weight)

        self.assertIsNone(gene, "do not create gene that already exits")

    def test_initializing_nn__three_inputs_one_outputs(self):
        workplace = Workplace(3, 1, 1)
        worker = Worker(workplace)

        # nn, counter, history = Worker.initialize_nn(Workplace.N_INPUT, Workplace.N_OUTPUT,
        #                                             Workplace.INNOV_COUNTER, Workplace.INNOV_HISTORY)
        #
        # self.assertEqual(counter, 2, "incremented innovation counter check")
        #
        # # in, out, weight, enabled, innov
        # #  0,   3,     w1,       1,     0,
        # #  1,   3,     w2,       1,     1,
        # #  2,   3,     w3,       1,     2,
        # self.assertEqual(nn.connect_genes.shape, (3, 5), "shape of connect genes")
        #
        # self.assertEqual(nn.connect_genes[0, 0], 0, "input")
        # self.assertEqual(nn.connect_genes[0, 1], 3, "output")
        # self.assertIsInstance(nn.connect_genes[0, 2], float, "weight")
        # self.assertEqual(nn.connect_genes[0, 3], 1, "enabled")
        # self.assertEqual(nn.connect_genes[0, 4], 0, "innovation number")
        #
        # self.assertEqual(nn.connect_genes[1, 0], 1, "input")
        # self.assertEqual(nn.connect_genes[1, 1], 3, "output")
        # self.assertIsInstance(nn.connect_genes[1, 2], float, "weight")
        # self.assertEqual(nn.connect_genes[1, 3], 1, "enabled")
        # self.assertEqual(nn.connect_genes[1, 4], 1, "innovation number")
        #
        # self.assertEqual(nn.connect_genes[2, 0], 2, "input")
        # self.assertEqual(nn.connect_genes[2, 1], 3, "output")
        # self.assertIsInstance(nn.connect_genes[2, 2], float, "weight")
        # self.assertEqual(nn.connect_genes[2, 3], 1, "enabled")
        # self.assertEqual(nn.connect_genes[2, 4], 2, "innovation number")

    # def test_initializing_nn__two_inputs_three_outputs(self):
    #     ori_nn = NeuralNetwork(2, 3)
    #     nn, counter, history = worker.initialize(ori_nn, workplace.INNOV_COUNTER, workplace.INNOV_HISTORY)
    #     # in, out, weight, enabled, innov
    #     #  0,   2,     w1,       1,     0,
    #     #  0,   3,     w2,       1,     1,
    #     #  0,   4,     w3,       1,     2,
    #     #  1,   2,     w4,       1,     3,
    #     #  1,   3,     w5,       1,     4,
    #     #  1,   4,     w6,       1,     5,
    #     self.assertEqual(nn.connect_genes.shape, (6, 5), "shape of connect genes")
    #
    #     self.assertEqual(nn.connect_genes[0, 0], 0, "input")
    #     self.assertEqual(nn.connect_genes[0, 1], 2, "output")
    #     self.assertIsInstance(nn.connect_genes[0, 2], float, "weight")
    #     self.assertEqual(nn.connect_genes[0, 3], 1, "enabled")
    #     self.assertEqual(nn.connect_genes[0, 4], 0, "innovation number")
    #
    #     self.assertEqual(nn.connect_genes[1, 0], 0, "input")
    #     self.assertEqual(nn.connect_genes[1, 1], 3, "output")
    #     self.assertIsInstance(nn.connect_genes[1, 2], float, "weight")
    #     self.assertEqual(nn.connect_genes[1, 3], 1, "enabled")
    #     self.assertEqual(nn.connect_genes[1, 4], 1, "innovation number")
    #
    #     self.assertEqual(nn.connect_genes[2, 0], 0, "input")
    #     self.assertEqual(nn.connect_genes[2, 1], 4, "output")
    #     self.assertIsInstance(nn.connect_genes[2, 2], float, "weight")
    #     self.assertEqual(nn.connect_genes[2, 3], 1, "enabled")
    #     self.assertEqual(nn.connect_genes[2, 4], 2, "innovation number")
    #
    #     self.assertEqual(nn.connect_genes[3, 0], 1, "input")
    #     self.assertEqual(nn.connect_genes[3, 1], 2, "output")
    #     self.assertIsInstance(nn.connect_genes[3, 2], float, "weight")
    #     self.assertEqual(nn.connect_genes[3, 3], 1, "enabled")
    #     self.assertEqual(nn.connect_genes[3, 4], 3, "innovation number")
    #
    #     self.assertEqual(nn.connect_genes[4, 0], 1, "input")
    #     self.assertEqual(nn.connect_genes[4, 1], 3, "output")
    #     self.assertIsInstance(nn.connect_genes[4, 2], float, "weight")
    #     self.assertEqual(nn.connect_genes[4, 3], 1, "enabled")
    #     self.assertEqual(nn.connect_genes[4, 4], 4, "innovation number")
    #
    #     self.assertEqual(nn.connect_genes[5, 0], 1, "input")
    #     self.assertEqual(nn.connect_genes[5, 1], 4, "output")
    #     self.assertIsInstance(nn.connect_genes[5, 2], float, "weight")
    #     self.assertEqual(nn.connect_genes[5, 3], 1, "enabled")
    #     self.assertEqual(nn.connect_genes[5, 4], 5, "innovation number")
    #
    # def test_innovation_counter__initializing_two_nn__three_inputs_one_output(self):
    #     nn1 = NeuralNetwork(3, 1)
    #     nn2 = NeuralNetwork(3, 1)
    #
    #     self.assertEqual(workplace.INNOV_COUNTER, -1)
    #     nn, counter, history = worker.initialize(nn1, workplace.INNOV_COUNTER, workplace.INNOV_HISTORY)
    #     workplace.INNOV_COUNTER = counter
    #     nn, counter2, history = worker.initialize(nn2, workplace.INNOV_COUNTER, workplace.INNOV_HISTORY)
    #
    #     self.assertEqual(workplace.INNOV_COUNTER, counter2, "building same topology do not increase innov counter")
    #
    # def test_innovation_history__initializing_two_nn__three_inputs_one_output(self):
    #     nn1 = NeuralNetwork(3, 1)
    #     nn2 = NeuralNetwork(3, 1)
    #
    #     self.assertIsNone(workplace.INNOV_HISTORY, "no history yet")
    #     nn, counter, history = worker.initialize(nn1, workplace.INNOV_COUNTER, workplace.INNOV_HISTORY)
    #     workplace.INNOV_HISTORY = history
    #     nn, counter, history2 = worker.initialize(nn2, workplace.INNOV_COUNTER, workplace.INNOV_HISTORY)
    #     self.assertTrue(np.array_equal(workplace.INNOV_COUNTER, history2),
    #                     "building same topology do not change history")
    #
    #     # in, out, weight, enabled, innov
    #     #  0,   3,      0,       1,     0,
    #     #  1,   3,      0,       1,     1,
    #     #  2,   3,      0,       1,     2,
    #     self.assertEqual(history.shape, (3, 5), "shape of history")
    #
    #     self.assertEqual(nn.connect_genes[0, 0], 0, "input")
    #     self.assertEqual(nn.connect_genes[0, 1], 3, "output")
    #     self.assertEqual(nn.connect_genes[0, 2], 0, "weight")
    #     self.assertEqual(nn.connect_genes[0, 3], 1, "enabled")
    #     self.assertEqual(nn.connect_genes[0, 4], 0, "innovation number")
    #
    #     self.assertEqual(nn.connect_genes[1, 0], 1, "input")
    #     self.assertEqual(nn.connect_genes[1, 1], 3, "output")
    #     self.assertEqual(nn.connect_genes[1, 2], 0, "weight")
    #     self.assertEqual(nn.connect_genes[1, 3], 1, "enabled")
    #     self.assertEqual(nn.connect_genes[1, 4], 1, "innovation number")
    #
    #     self.assertEqual(nn.connect_genes[2, 0], 2, "input")
    #     self.assertEqual(nn.connect_genes[2, 1], 3, "output")
    #     self.assertEqual(nn.connect_genes[2, 2], 0, "weight")
    #     self.assertEqual(nn.connect_genes[2, 3], 1, "enabled")
    #     self.assertEqual(nn.connect_genes[2, 4], 2, "innovation number")





    def test_add_node(self):
        pass


    def test_mutate_connection(self):
        pass

    def test_mutate_weight(self):
        pass

    def test_crossover(self):
        pass
