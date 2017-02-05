import unittest
import numpy as np
from players.neuralnet import NeuralNetwork
from players.worker import Worker
from globaladmin.workplace import Workplace
from players.activation import sigmoid


class WorkerTest(unittest.TestCase):

    def test_is_connect_exist_nn__no_gene(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)
        nn = NeuralNetwork()
        nn.connect_genes = None

        node_in = 0
        node_out = 6
        self.assertFalse(worker.is_connect_exist_nn(node_in, node_out, nn))

    def test_is_connect_exist_nn__yes_gene_no_connect(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)
        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 6, 0, 1, 1]])

        node_in = 0
        node_out = 7
        self.assertFalse(worker.is_connect_exist_nn(node_in, node_out, nn))

    def test_is_connect_exist_nn__yes_gene_yes_connect(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)
        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 6, 0, 1, 1]])

        node_in = 0
        node_out = 6
        self.assertTrue(worker.is_connect_exist_nn(node_in, node_out, nn))

    def test_is_connect_exist_nn__yes(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)
        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 6, 0, 1, 1]])

        node_in = 0
        node_out = 6
        self.assertTrue(worker.is_connect_exist_nn(node_in, node_out, nn))

    def test_is_connect_exist_global__no_history(self):
        workplace = Workplace(n_input=3, n_output=1)
        workplace.innov_history = {}
        worker = Worker(workplace)

        node_in = 1
        node_out = 3

        self.assertEqual(worker.is_connect_exist_global(node_in, node_out), None)

    def test_is_connect_exist_global__yes_history_yes_match(self):
        workplace = Workplace(n_input=3, n_output=1)
        workplace.innov_history = {(1, 3): 1, (2, 3): 4}
        worker = Worker(workplace)

        node_in = 2
        node_out = 3

        self.assertEqual(worker.is_connect_exist_global(node_in, node_out), 4)

    def test_is_connect_exist_global__yes_history_no_match(self):
        workplace = Workplace(n_input=3, n_output=1)
        workplace.innov_history = {(1, 3): 1, (2, 3): 4}
        worker = Worker(workplace)

        node_in = 0
        node_out = 3

        self.assertEqual(worker.is_connect_exist_global(node_in, node_out), None)

    def test_is_bias_node__yes(self):
        workplace = Workplace(n_input=3, n_output=1)
        worker = Worker(workplace)

        node = 0
        self.assertTrue(worker.is_bias_node(node))

    def test_is_bias_node__no(self):
        workplace = Workplace(n_input=3, n_output=1)
        worker = Worker(workplace)

        node = 1
        self.assertFalse(worker.is_bias_node(node))

    def test_is_input_node__yes(self):
        workplace = Workplace(n_input=3, n_output=1)
        worker = Worker(workplace)

        node = 1
        self.assertTrue(worker.is_input_node(node))
        node = 2
        self.assertTrue(worker.is_input_node(node))
        node = 3
        self.assertTrue(worker.is_input_node(node))

    def test_is_input_node__no(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        node = 0
        self.assertFalse(worker.is_input_node(node))
        node = 6
        self.assertFalse(worker.is_input_node(node))
        node = 7
        self.assertFalse(worker.is_input_node(node))
        node = 10
        self.assertFalse(worker.is_input_node(node))

    def test_is_output_node__yes(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        node = 6
        self.assertTrue(worker.is_output_node(node))
        node = 7
        self.assertTrue(worker.is_output_node(node))
        node = 8
        self.assertTrue(worker.is_output_node(node))
        node = 9
        self.assertTrue(worker.is_output_node(node))

    def test_is_output_node__no(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        node = 0
        self.assertFalse(worker.is_output_node(node))
        node = 3
        self.assertFalse(worker.is_output_node(node))
        node = 10
        self.assertFalse(worker.is_output_node(node))
        node = 20
        self.assertFalse(worker.is_output_node(node))

    def test_is_in_in_connect__yes(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        node1 = 1
        node2 = 2
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 4
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 3
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 2
        node2 = 4
        self.assertTrue(worker.is_in_in_connect(node1, node2))

    def test_is_in_in_connect__no(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        node1 = 0
        node2 = 2
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 0
        node2 = 3
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 7
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 2
        node2 = 8
        self.assertFalse(worker.is_in_in_connect(node1, node2))

    def test_is_out_out_connect__yes(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        node1 = 6
        node2 = 7
        self.assertTrue(worker.is_out_out_connect(node1, node2))
        node1 = 6
        node2 = 8
        self.assertTrue(worker.is_out_out_connect(node1, node2))
        node1 = 7
        node2 = 8
        self.assertTrue(worker.is_out_out_connect(node1, node2))
        node1 = 8
        node2 = 9
        self.assertTrue(worker.is_out_out_connect(node1, node2))

    def test_is_out_out_connect__no(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        node1 = 0
        node2 = 6
        self.assertFalse(worker.is_out_out_connect(node1, node2))
        node1 = 0
        node2 = 7
        self.assertFalse(worker.is_out_out_connect(node1, node2))
        node1 = 1
        node2 = 7
        self.assertFalse(worker.is_out_out_connect(node1, node2))
        node1 = 2
        node2 = 8
        self.assertFalse(worker.is_out_out_connect(node1, node2))

    def test_is_recursive_connect__yes(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        node1 = 5
        node2 = 5
        self.assertTrue(worker.is_recursive_connect(node1, node2))
        node1 = 7
        node2 = 7
        self.assertTrue(worker.is_recursive_connect(node1, node2))

    def test_is_recursive_connect__no(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        node1 = 5
        node2 = 1
        self.assertFalse(worker.is_recursive_connect(node1, node2))
        node1 = 7
        node2 = 2
        self.assertFalse(worker.is_recursive_connect(node1, node2))

    def test_increment_innov_counter(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        self.assertEqual(workplace.innov_counter, -1)

        worker.increment_innov_counter()
        self.assertEqual(workplace.innov_counter, 0)
        worker.increment_innov_counter()
        self.assertEqual(workplace.innov_counter, 1)
        worker.increment_innov_counter()
        self.assertEqual(workplace.innov_counter, 2)

    def test_record_innov_history__connect_already_exist(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)
        workplace.innov_history[(1, 6)] = 4

        connect = (1, 6)
        self.assertRaises(AssertionError, worker.record_innov_history, connect)

    def test_record_innov_history__ok(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)
        workplace.innov_counter = 5
        workplace.innov_history[(1, 6)] = 4

        connect = (1, 5)
        worker.record_innov_history(connect)
        self.assertEqual(workplace.innov_history, {(1, 6): 4, (1, 5): 5})

    def test_add_connect(self):
        workplace = Workplace(n_input=5, n_output=4)
        worker = Worker(workplace)

        nn = NeuralNetwork()
        worker.add_connect(node_in=0,
                           node_out=6,
                           weight=0.0,
                           nn=nn)

        self.assertTrue(np.array_equal(nn.connect_genes, np.array([[0, 6, 0, 1, 0]])))
        self.assertEqual(workplace.innov_counter, 0, "innov counter should be 0")
        self.assertEqual(workplace.innov_history, {(0, 6): 0})

        worker.add_connect(node_in=0,
                           node_out=7,
                           weight=1.0,
                           nn=nn)

        self.assertTrue(np.array_equal(nn.connect_genes, np.array([[0, 6, 0, 1, 0],
                                                                   [0, 7, 1, 1, 1]])))
        self.assertEqual(workplace.innov_counter, 1, "innov counter should be 1")
        self.assertEqual(workplace.innov_history, {(0, 6): 0,
                                                   (0, 7): 1})

    def test_create_initial_info__3_inputs_1_output(self):
        # TODO: bias should be added
        workplace = Workplace(n_input=3, n_output=1)
        worker = Worker(workplace)

        initial_genes, initial_history, counter = worker.create_initial_info()

        # innov_counter check
        self.assertEqual(counter, 2, "incremented innovation counter check")

        # in, out, weight, enabled, innov
        #  0,   3,     w1,       1,     0,
        #  1,   3,     w2,       1,     1,
        #  2,   3,     w3,       1,     2,
        self.assertEqual(initial_genes.shape, (3, 5), "shape of connect genes")

        self.assertEqual(initial_genes[0, 0], 0, "input")
        self.assertEqual(initial_genes[0, 1], 3, "output")
        self.assertIsInstance(initial_genes[0, 2], float, "weight")
        self.assertEqual(initial_genes[0, 3], 1, "enabled")
        self.assertEqual(initial_genes[0, 4], 0, "innovation number")

        self.assertEqual(initial_genes[1, 0], 1, "input")
        self.assertEqual(initial_genes[1, 1], 3, "output")
        self.assertIsInstance(initial_genes[1, 2], float, "weight")
        self.assertEqual(initial_genes[1, 3], 1, "enabled")
        self.assertEqual(initial_genes[1, 4], 1, "innovation number")

        self.assertEqual(initial_genes[2, 0], 2, "input")
        self.assertEqual(initial_genes[2, 1], 3, "output")
        self.assertIsInstance(initial_genes[2, 2], float, "weight")
        self.assertEqual(initial_genes[2, 3], 1, "enabled")
        self.assertEqual(initial_genes[2, 4], 2, "innovation number")

        # innov_history check
        self.assertEqual(initial_history, {(0, 3): 0,
                                           (1, 3): 1,
                                           (2, 3): 2})

    def test_create_initial_info__2_inputs_3_output(self):
        # TODO: bias should be added
        workplace = Workplace(n_input=2, n_output=3)
        worker = Worker(workplace)

        initial_genes, initial_history, counter = worker.create_initial_info()

        # innov_counter check
        self.assertEqual(counter, 5, "incremented innovation counter check")

        # in, out, weight, enabled, innov
        #  0,   2,     w1,       1,     0,
        #  0,   3,     w2,       1,     1,
        #  0,   4,     w3,       1,     2,
        #  1,   2,     w4,       1,     3,
        #  1,   3,     w5,       1,     4,
        #  1,   4,     w6,       1,     5,
        self.assertEqual(initial_genes.shape, (6, 5), "shape of connect genes")

        self.assertEqual(initial_genes[0, 0], 0, "input")
        self.assertEqual(initial_genes[0, 1], 2, "output")
        self.assertIsInstance(initial_genes[0, 2], float, "weight")
        self.assertEqual(initial_genes[0, 3], 1, "enabled")
        self.assertEqual(initial_genes[0, 4], 0, "innovation number")

        self.assertEqual(initial_genes[1, 0], 0, "input")
        self.assertEqual(initial_genes[1, 1], 3, "output")
        self.assertIsInstance(initial_genes[1, 2], float, "weight")
        self.assertEqual(initial_genes[1, 3], 1, "enabled")
        self.assertEqual(initial_genes[1, 4], 1, "innovation number")

        self.assertEqual(initial_genes[2, 0], 0, "input")
        self.assertEqual(initial_genes[2, 1], 4, "output")
        self.assertIsInstance(initial_genes[2, 2], float, "weight")
        self.assertEqual(initial_genes[2, 3], 1, "enabled")
        self.assertEqual(initial_genes[2, 4], 2, "innovation number")

        self.assertEqual(initial_genes[3, 0], 1, "input")
        self.assertEqual(initial_genes[3, 1], 2, "output")
        self.assertIsInstance(initial_genes[3, 2], float, "weight")
        self.assertEqual(initial_genes[3, 3], 1, "enabled")
        self.assertEqual(initial_genes[3, 4], 3, "innovation number")

        self.assertEqual(initial_genes[4, 0], 1, "input")
        self.assertEqual(initial_genes[4, 1], 3, "output")
        self.assertIsInstance(initial_genes[4, 2], float, "weight")
        self.assertEqual(initial_genes[4, 3], 1, "enabled")
        self.assertEqual(initial_genes[4, 4], 4, "innovation number")

        self.assertEqual(initial_genes[5, 0], 1, "input")
        self.assertEqual(initial_genes[5, 1], 4, "output")
        self.assertIsInstance(initial_genes[5, 2], float, "weight")
        self.assertEqual(initial_genes[5, 3], 1, "enabled")
        self.assertEqual(initial_genes[5, 4], 5, "innovation number")

        # innov_history check
        self.assertEqual(initial_history, {(0, 2): 0,
                                           (0, 3): 1,
                                           (0, 4): 2,
                                           (1, 2): 3,
                                           (1, 3): 4,
                                           (1, 4): 5})

    def test_initialize_nns__10_nns(self):
        workplace = Workplace(n_input=3, n_output=1, n_nn=10)
        worker = Worker(workplace)

        worker.initialize_workplace()

        self.assertEqual(len(workplace.nns), 10)
        self.assertEqual(workplace.innov_history, {(0, 4): 0,
                                                   (1, 4): 1,
                                                   (2, 4): 2,
                                                   (3, 4): 3})
        self.assertEqual(workplace.innov_counter, 3)

        # test if two nns have identical genes except weight
        nn1 = workplace.nns[0]
        nn2 = workplace.nns[9]
        gene1 = nn1.connect_genes
        gene2 = nn2.connect_genes
        gene1_w_removed = np.delete(gene1, 3, 1)
        gene2_w_removed = np.delete(gene2, 3, 1)
        self.assertEqual(gene1.shape, (3, 5))
        self.assertFalse(np.array_equal(gene1, gene2), "two genes must have different weights")
        self.assertTrue(np.array_equal(gene1_w_removed, gene2_w_removed), "two genes have identical other elements")

    def test_activate__valid_input(self):
        bias = 2
        workplace = Workplace(3, 4, bias=bias)
        worker = Worker(workplace)

        xs = np.array([[1, 2, 3]])
        ws = np.array([[1], [1], [1]])
        y = worker.activate(xs, ws)

        self.assertEqual(y, sigmoid(np.dot(xs, ws)) - bias)

    def test_activate__ws_xs_size_mismatch(self):
        bias = 2
        workplace = Workplace(3, 4, bias=bias)
        worker = Worker(workplace)

        xs = np.array([[1, 2, 3, 4]])
        ws = np.array([[1], [1], [1]])

        self.assertRaises(AssertionError, worker.activate, xs, ws)

    def test_activate__xs_invalid_shape(self):
        bias = 2
        workplace = Workplace(3, 4, bias=bias)
        worker = Worker(workplace)

        xs = np.array([[1], [1], [1]])
        ws = np.array([[1, 2, 3]])

        self.assertRaises(AssertionError, worker.activate, xs, ws)

    def test_feedforward__AND(self):
        workplace = Workplace(2, 1, bias=0.2)
        worker = Worker(workplace)

        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 2, 0.5, 1, 0],
                                     [1, 2, 0.5, 1, 1]])

        input_data = np.array([0, 0])
        self.assertTrue(worker.feedforward(input_data) < 0.5)
        input_data = np.array([0, 1])
        self.assertTrue(worker.feedforward(input_data) < 0.5)
        input_data = np.array([1, 0])
        self.assertTrue(worker.feedforward(input_data) < 0.5)
        input_data = np.array([1, 1])
        self.assertTrue(worker.feedforward(input_data) > 0.5)

    def test_feedforward__OR(self):
        workplace = Workplace(2, 1, bias=0.1)
        worker = Worker(workplace)

        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 2, 0.5, 1, 0],
                                     [1, 2, 0.5, 1, 1]])

        input_data = np.array([0, 0])
        self.assertTrue(worker.feedforward(input_data) < 0.5)
        input_data = np.array([0, 1])
        self.assertTrue(worker.feedforward(input_data) > 0.5)
        input_data = np.array([1, 0])
        self.assertTrue(worker.feedforward(input_data) > 0.5)
        input_data = np.array([1, 1])
        self.assertTrue(worker.feedforward(input_data) > 0.5)

    def test_feedforward__XOR(self):
        workplace = Workplace(2, 1, bias=2.5)
        worker = Worker(workplace)

        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 2, 0.5, 1, 0],
                                     [1, 2, 0.5, 1, 1],
                                     [0, 3, 0.5, 1, 2],
                                     [3, 2, 0.5, 1, 3],
                                     [1, 3, 0.5, 1, 4]])

        input_data = np.array([0, 0])
        self.assertTrue(worker.feedforward(input_data) < 0.5)
        input_data = np.array([0, 1])
        self.assertTrue(worker.feedforward(input_data) > 0.5)
        input_data = np.array([1, 0])
        self.assertTrue(worker.feedforward(input_data) > 0.5)
        input_data = np.array([1, 1])
        self.assertTrue(worker.feedforward(input_data) < 0.5)




    #
    #
    #
    # def test_create_connection_gene__no_history(self):
    #     workplace = Workplace(n_input=3, n_output=1)
    #     worker = Worker(workplace)
    #
    #     node_in = 1
    #     node_out = 3
    #     weight = 0.5
    #     history = np.empty((1, 3))
    #
    #     gene = worker.create_connection_gene(node_in, node_out, weight, history)
    #
    #     self.assertTrue(np.array_equal(gene, np.array([node_in, node_out, weight, 1, 0])))
    #
    # def test_add_connect_gene__yes_history(self):
    #     workplace = Workplace(3, 1)
    #     worker = Worker(workplace)
    #     node_in = 1
    #     node_out = 3
    #     weight = 0.5
    #     workplace.innov_history = [[1, 3]]
    #
    #     gene = worker.create_connection_gene(node_in, node_out, weight)
    #
    #     self.assertIsNone(gene, "do not create gene that already exits")


    def test_add_node(self):
        pass


    def test_mutate_connection(self):
        pass

    def test_mutate_weight(self):
        pass

    def test_crossover(self):
        pass
