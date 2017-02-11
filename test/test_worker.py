import unittest
import numpy as np
from players.neuralnet import NeuralNetwork
from players.worker import Worker
from globaladmin.workplace import Workplace
from players.activation import sigmoid, linear


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

    def test_is_bias_node__yes_bias_yes(self):
        workplace = Workplace(n_input=3, n_output=1, bias=1)
        worker = Worker(workplace)

        node = 0
        self.assertTrue(worker.is_bias_node(node))

    def test_is_bias_node__yes_bias_no(self):
        workplace = Workplace(n_input=3, n_output=1)
        worker = Worker(workplace)

        node = 1
        self.assertFalse(worker.is_bias_node(node))

    def test_is_bias_node__no_bias_no(self):
        workplace = Workplace(n_input=3, n_output=1, bias=None)
        worker = Worker(workplace)

        node = 0
        self.assertFalse(worker.is_bias_node(node))

    def test_is_input_node__yes_bias_yes(self):
        workplace = Workplace(n_input=3, n_output=1, bias=1)
        worker = Worker(workplace)

        node = 1
        self.assertTrue(worker.is_input_node(node))
        node = 2
        self.assertTrue(worker.is_input_node(node))
        node = 3
        self.assertTrue(worker.is_input_node(node))

    def test_is_input_node__yes_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
        worker = Worker(workplace)

        node = 0
        self.assertFalse(worker.is_input_node(node))
        node = 6
        self.assertFalse(worker.is_input_node(node))
        node = 7
        self.assertFalse(worker.is_input_node(node))
        node = 10
        self.assertFalse(worker.is_input_node(node))

    def test_is_input_node__no_bias_yes(self):
        workplace = Workplace(n_input=3, n_output=1, bias=None)
        worker = Worker(workplace)

        node = 0
        self.assertTrue(worker.is_input_node(node))
        node = 1
        self.assertTrue(worker.is_input_node(node))
        node = 2
        self.assertTrue(worker.is_input_node(node))

    def test_is_input_node__no_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=None)
        worker = Worker(workplace)

        node = 5
        self.assertFalse(worker.is_input_node(node))
        node = 6
        self.assertFalse(worker.is_input_node(node))
        node = 7
        self.assertFalse(worker.is_input_node(node))
        node = 10
        self.assertFalse(worker.is_input_node(node))

    def test_is_output_node__yes_bias_yes(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
        worker = Worker(workplace)

        node = 6
        self.assertTrue(worker.is_output_node(node))
        node = 7
        self.assertTrue(worker.is_output_node(node))
        node = 8
        self.assertTrue(worker.is_output_node(node))
        node = 9
        self.assertTrue(worker.is_output_node(node))

    def test_is_output_node__yes_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
        worker = Worker(workplace)

        node = 0
        self.assertFalse(worker.is_output_node(node))
        node = 3
        self.assertFalse(worker.is_output_node(node))
        node = 5
        self.assertFalse(worker.is_output_node(node))
        node = 10
        self.assertFalse(worker.is_output_node(node))
        node = 20
        self.assertFalse(worker.is_output_node(node))

    def test_is_output_node__no_bias_yes(self):
        workplace = Workplace(n_input=5, n_output=4, bias=None)
        worker = Worker(workplace)

        node = 5
        self.assertTrue(worker.is_output_node(node))
        node = 6
        self.assertTrue(worker.is_output_node(node))
        node = 7
        self.assertTrue(worker.is_output_node(node))
        node = 8
        self.assertTrue(worker.is_output_node(node))

    def test_is_output_node__no_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=None)
        worker = Worker(workplace)

        node = 0
        self.assertFalse(worker.is_output_node(node))
        node = 3
        self.assertFalse(worker.is_output_node(node))
        node = 4
        self.assertFalse(worker.is_output_node(node))
        node = 9
        self.assertFalse(worker.is_output_node(node))
        node = 20
        self.assertFalse(worker.is_output_node(node))

    def test_is_bias_in_connect__yes_bias_yes(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
        worker = Worker(workplace)

        node1 = 0
        node2 = 1
        self.assertTrue(worker.is_bias_in_connect(node1, node2))
        node1 = 0
        node2 = 3
        self.assertTrue(worker.is_bias_in_connect(node1, node2))
        node1 = 0
        node2 = 5
        self.assertTrue(worker.is_bias_in_connect(node1, node2))

    def test_is_bias_in_connect__yes_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
        worker = Worker(workplace)

        node1 = 0
        node2 = 6
        self.assertFalse(worker.is_bias_in_connect(node1, node2))
        node1 = 0
        node2 = 7
        self.assertFalse(worker.is_bias_in_connect(node1, node2))
        node1 = 0
        node2 = 20
        self.assertFalse(worker.is_bias_in_connect(node1, node2))
        node1 = 1
        node2 = 2
        self.assertFalse(worker.is_bias_in_connect(node1, node2))

    def test_is_bias_in_connect__no_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=None)
        worker = Worker(workplace)

        node1 = 0
        node2 = 1
        self.assertFalse(worker.is_bias_in_connect(node1, node2))
        node1 = 0
        node2 = 2
        self.assertFalse(worker.is_bias_in_connect(node1, node2))
        node1 = 0
        node2 = 3
        self.assertFalse(worker.is_bias_in_connect(node1, node2))
        node1 = 1
        node2 = 2
        self.assertFalse(worker.is_bias_in_connect(node1, node2))
        node1 = 0
        node2 = 20
        self.assertFalse(worker.is_bias_in_connect(node1, node2))

    def test_is_in_in_connect__yes_bias_yes(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
        worker = Worker(workplace)

        node1 = 1
        node2 = 2
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 3
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 4
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 5
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 2
        node2 = 5
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 3
        node2 = 4
        self.assertTrue(worker.is_in_in_connect(node1, node2))

    def test_is_in_in_connect__yes_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
        worker = Worker(workplace)

        node1 = 0
        node2 = 1
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 0
        node2 = 2
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 0
        node2 = 3
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 6
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 7
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 5
        node2 = 6
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 2
        node2 = 8
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 2
        node2 = 0
        self.assertFalse(worker.is_in_in_connect(node1, node2))

    def test_is_in_in_connect__no_bias_yes(self):
        workplace = Workplace(n_input=5, n_output=4, bias=None)
        worker = Worker(workplace)

        node1 = 0
        node2 = 1
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 0
        node2 = 2
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 0
        node2 = 3
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 0
        node2 = 4
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 4
        self.assertTrue(worker.is_in_in_connect(node1, node2))
        node1 = 2
        node2 = 4
        self.assertTrue(worker.is_in_in_connect(node1, node2))

    def test_is_in_in_connect__no_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=None)
        worker = Worker(workplace)

        node1 = 0
        node2 = 5
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 5
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 2
        node2 = 5
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 6
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 1
        node2 = 7
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 4
        node2 = 6
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 2
        node2 = 8
        self.assertFalse(worker.is_in_in_connect(node1, node2))
        node1 = 8
        node2 = 0
        self.assertFalse(worker.is_in_in_connect(node1, node2))

    def test_is_out_out_connect__yes_bias_yes(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
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

    def test_is_out_out_connect__yes_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
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

    def test_is_out_out_connect__no_bias_yes(self):
        workplace = Workplace(n_input=5, n_output=4, bias=None)
        worker = Worker(workplace)

        node1 = 5
        node2 = 6
        self.assertTrue(worker.is_out_out_connect(node1, node2))
        node1 = 5
        node2 = 8
        self.assertTrue(worker.is_out_out_connect(node1, node2))
        node1 = 6
        node2 = 8
        self.assertTrue(worker.is_out_out_connect(node1, node2))
        node1 = 7
        node2 = 8
        self.assertTrue(worker.is_out_out_connect(node1, node2))

    def test_is_out_out_connect__no_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=None)
        worker = Worker(workplace)

        node1 = 4
        node2 = 5
        self.assertFalse(worker.is_out_out_connect(node1, node2))
        node1 = 0
        node2 = 7
        self.assertFalse(worker.is_out_out_connect(node1, node2))
        node1 = 1
        node2 = 7
        self.assertFalse(worker.is_out_out_connect(node1, node2))
        node1 = 5
        node2 = 9
        self.assertFalse(worker.is_out_out_connect(node1, node2))

    def test_is_in_bias_at_end_connect__yes_bias_yes(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
        worker = Worker(workplace)

        node1 = 6
        node2 = 0
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 7
        node2 = 0
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 1
        node2 = 0
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 10
        node2 = 0
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 6
        node2 = 1
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 6
        node2 = 2
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 6
        node2 = 3
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 6
        node2 = 4
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 6
        node2 = 5
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))

    def test_is_in_bias_at_end_connect__yes_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=1)
        worker = Worker(workplace)

        node1 = 0
        node2 = 6
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 0
        node2 = 7
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 1
        node2 = 6
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 10
        node2 = 6
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 3
        node2 = 6
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 2
        node2 = 6
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 9
        node2 = 6
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))

    def test_is_in_bias_at_end_connect__no_bias_yes(self):
        workplace = Workplace(n_input=5, n_output=4, bias=None)
        worker = Worker(workplace)

        node1 = 6
        node2 = 0
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 7
        node2 = 0
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 1
        node2 = 0
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 10
        node2 = 0
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 6
        node2 = 1
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 6
        node2 = 2
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 6
        node2 = 3
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 6
        node2 = 4
        self.assertTrue(worker.is_in_bias_at_end_connect(node1, node2))

    def test_is_in_bias_at_end_connect__no_bias_no(self):
        workplace = Workplace(n_input=5, n_output=4, bias=None)
        worker = Worker(workplace)

        node1 = 0
        node2 = 5
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 0
        node2 = 6
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 1
        node2 = 7
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 10
        node2 = 5
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 3
        node2 = 5
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 2
        node2 = 5
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 9
        node2 = 5
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))
        node1 = 6
        node2 = 5
        self.assertFalse(worker.is_in_bias_at_end_connect(node1, node2))

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

    def test_initialize_nn__3_inputs_1_output_yes_bias(self):
        workplace = Workplace(n_input=3, n_output=1, bias=1)
        worker = Worker(workplace)
        nn = NeuralNetwork()

        worker.initialize_nn(nn)

        # innov_counter check
        self.assertEqual(workplace.innov_counter, 3, "incremented innovation counter check")

        # in, out, weight, enabled, innov
        #  0,   4,     w1,       1,     0,
        #  1,   4,     w2,       1,     1,
        #  2,   4,     w3,       1,     2,
        #  3,   4,     w4,       1,     3,
        self.assertEqual(nn.connect_genes.shape, (4, 5), "shape of connect genes")

        self.assertEqual(nn.connect_genes[0, 0], 0, "input")
        self.assertEqual(nn.connect_genes[0, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[0, 2], float, "weight")
        self.assertEqual(nn.connect_genes[0, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[0, 4], 0, "innovation number")

        self.assertEqual(nn.connect_genes[1, 0], 1, "input")
        self.assertEqual(nn.connect_genes[1, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[1, 2], float, "weight")
        self.assertEqual(nn.connect_genes[1, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[1, 4], 1, "innovation number")

        self.assertEqual(nn.connect_genes[2, 0], 2, "input")
        self.assertEqual(nn.connect_genes[2, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[2, 2], float, "weight")
        self.assertEqual(nn.connect_genes[2, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[2, 4], 2, "innovation number")

        self.assertEqual(nn.connect_genes[3, 0], 3, "input")
        self.assertEqual(nn.connect_genes[3, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[3, 2], float, "weight")
        self.assertEqual(nn.connect_genes[3, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[3, 4], 3, "innovation number")

        # innov_history check
        self.assertEqual(workplace.innov_history, {(0, 4): 0,
                                                   (1, 4): 1,
                                                   (2, 4): 2,
                                                   (3, 4): 3})

    def test_initialize_nn__3_inputs_1_output_no_bias(self):
        workplace = Workplace(n_input=3, n_output=1, bias=None)
        worker = Worker(workplace)
        nn = NeuralNetwork()

        worker.initialize_nn(nn)

        # innov_counter check
        self.assertEqual(workplace.innov_counter, 2, "incremented innovation counter check")

        # in, out, weight, enabled, innov
        #  0,   3,     w1,       1,     0,
        #  1,   3,     w2,       1,     1,
        #  2,   3,     w3,       1,     2,
        self.assertEqual(nn.connect_genes.shape, (3, 5), "shape of connect genes")

        self.assertEqual(nn.connect_genes[0, 0], 0, "input")
        self.assertEqual(nn.connect_genes[0, 1], 3, "output")
        self.assertIsInstance(nn.connect_genes[0, 2], float, "weight")
        self.assertEqual(nn.connect_genes[0, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[0, 4], 0, "innovation number")

        self.assertEqual(nn.connect_genes[1, 0], 1, "input")
        self.assertEqual(nn.connect_genes[1, 1], 3, "output")
        self.assertIsInstance(nn.connect_genes[1, 2], float, "weight")
        self.assertEqual(nn.connect_genes[1, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[1, 4], 1, "innovation number")

        self.assertEqual(nn.connect_genes[2, 0], 2, "input")
        self.assertEqual(nn.connect_genes[2, 1], 3, "output")
        self.assertIsInstance(nn.connect_genes[2, 2], float, "weight")
        self.assertEqual(nn.connect_genes[2, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[2, 4], 2, "innovation number")

        # innov_history check
        self.assertEqual(workplace.innov_history, {(0, 3): 0,
                                                   (1, 3): 1,
                                                   (2, 3): 2})

    def test_initialize_nn__2_inputs_3_outputs_yes_bias(self):
        workplace = Workplace(n_input=2, n_output=3, bias=1)
        worker = Worker(workplace)
        nn = NeuralNetwork()

        worker.initialize_nn(nn)

        # innov_counter check
        self.assertEqual(workplace.innov_counter, 8, "incremented innovation counter check")

        # in, out, weight, enabled, innov
        #  0,   3,     w1,       1,     0,
        #  0,   4,     w2,       1,     1,
        #  0,   5,     w3,       1,     2,
        #  1,   3,     w4,       1,     3,
        #  1,   4,     w5,       1,     4,
        #  1,   5,     w6,       1,     5,
        #  2,   3,     w7,       1,     6,
        #  2,   4,     w8,       1,     7,
        #  2,   5,     w9,       1,     8,
        self.assertEqual(nn.connect_genes.shape, (9, 5), "shape of connect genes")

        self.assertEqual(nn.connect_genes[0, 0], 0, "input")
        self.assertEqual(nn.connect_genes[0, 1], 3, "output")
        self.assertIsInstance(nn.connect_genes[0, 2], float, "weight")
        self.assertEqual(nn.connect_genes[0, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[0, 4], 0, "innovation number")

        self.assertEqual(nn.connect_genes[1, 0], 0, "input")
        self.assertEqual(nn.connect_genes[1, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[1, 2], float, "weight")
        self.assertEqual(nn.connect_genes[1, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[1, 4], 1, "innovation number")

        self.assertEqual(nn.connect_genes[2, 0], 0, "input")
        self.assertEqual(nn.connect_genes[2, 1], 5, "output")
        self.assertIsInstance(nn.connect_genes[2, 2], float, "weight")
        self.assertEqual(nn.connect_genes[2, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[2, 4], 2, "innovation number")

        self.assertEqual(nn.connect_genes[3, 0], 1, "input")
        self.assertEqual(nn.connect_genes[3, 1], 3, "output")
        self.assertIsInstance(nn.connect_genes[3, 2], float, "weight")
        self.assertEqual(nn.connect_genes[3, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[3, 4], 3, "innovation number")

        self.assertEqual(nn.connect_genes[4, 0], 1, "input")
        self.assertEqual(nn.connect_genes[4, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[4, 2], float, "weight")
        self.assertEqual(nn.connect_genes[4, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[4, 4], 4, "innovation number")

        self.assertEqual(nn.connect_genes[5, 0], 1, "input")
        self.assertEqual(nn.connect_genes[5, 1], 5, "output")
        self.assertIsInstance(nn.connect_genes[5, 2], float, "weight")
        self.assertEqual(nn.connect_genes[5, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[5, 4], 5, "innovation number")

        self.assertEqual(nn.connect_genes[6, 0], 2, "input")
        self.assertEqual(nn.connect_genes[6, 1], 3, "output")
        self.assertIsInstance(nn.connect_genes[6, 2], float, "weight")
        self.assertEqual(nn.connect_genes[6, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[6, 4], 6, "innovation number")

        self.assertEqual(nn.connect_genes[7, 0], 2, "input")
        self.assertEqual(nn.connect_genes[7, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[7, 2], float, "weight")
        self.assertEqual(nn.connect_genes[7, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[7, 4], 7, "innovation number")

        self.assertEqual(nn.connect_genes[8, 0], 2, "input")
        self.assertEqual(nn.connect_genes[8, 1], 5, "output")
        self.assertIsInstance(nn.connect_genes[8, 2], float, "weight")
        self.assertEqual(nn.connect_genes[8, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[8, 4], 8, "innovation number")

        # innov_history check
        self.assertEqual(workplace.innov_history, {(0, 3): 0,
                                                   (0, 4): 1,
                                                   (0, 5): 2,
                                                   (1, 3): 3,
                                                   (1, 4): 4,
                                                   (1, 5): 5,
                                                   (2, 3): 6,
                                                   (2, 4): 7,
                                                   (2, 5): 8})

    def test_initialize_nn__2_inputs_3_outputs_no_bias(self):
        workplace = Workplace(n_input=2, n_output=3, bias=None)
        worker = Worker(workplace)
        nn = NeuralNetwork()

        worker.initialize_nn(nn)

        # innov_counter check
        self.assertEqual(workplace.innov_counter, 5, "incremented innovation counter check")

        # in, out, weight, enabled, innov
        #  0,   2,     w1,       1,     0,
        #  0,   3,     w2,       1,     1,
        #  0,   4,     w3,       1,     2,
        #  1,   2,     w4,       1,     3,
        #  1,   3,     w5,       1,     4,
        #  1,   4,     w6,       1,     5,
        self.assertEqual(nn.connect_genes.shape, (6, 5), "shape of connect genes")

        self.assertEqual(nn.connect_genes[0, 0], 0, "input")
        self.assertEqual(nn.connect_genes[0, 1], 2, "output")
        self.assertIsInstance(nn.connect_genes[0, 2], float, "weight")
        self.assertEqual(nn.connect_genes[0, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[0, 4], 0, "innovation number")

        self.assertEqual(nn.connect_genes[1, 0], 0, "input")
        self.assertEqual(nn.connect_genes[1, 1], 3, "output")
        self.assertIsInstance(nn.connect_genes[1, 2], float, "weight")
        self.assertEqual(nn.connect_genes[1, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[1, 4], 1, "innovation number")

        self.assertEqual(nn.connect_genes[2, 0], 0, "input")
        self.assertEqual(nn.connect_genes[2, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[2, 2], float, "weight")
        self.assertEqual(nn.connect_genes[2, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[2, 4], 2, "innovation number")

        self.assertEqual(nn.connect_genes[3, 0], 1, "input")
        self.assertEqual(nn.connect_genes[3, 1], 2, "output")
        self.assertIsInstance(nn.connect_genes[3, 2], float, "weight")
        self.assertEqual(nn.connect_genes[3, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[3, 4], 3, "innovation number")

        self.assertEqual(nn.connect_genes[4, 0], 1, "input")
        self.assertEqual(nn.connect_genes[4, 1], 3, "output")
        self.assertIsInstance(nn.connect_genes[4, 2], float, "weight")
        self.assertEqual(nn.connect_genes[4, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[4, 4], 4, "innovation number")

        self.assertEqual(nn.connect_genes[5, 0], 1, "input")
        self.assertEqual(nn.connect_genes[5, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[5, 2], float, "weight")
        self.assertEqual(nn.connect_genes[5, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[5, 4], 5, "innovation number")

        # innov_history check
        self.assertEqual(workplace.innov_history, {(0, 2): 0,
                                                   (0, 3): 1,
                                                   (0, 4): 2,
                                                   (1, 2): 3,
                                                   (1, 3): 4,
                                                   (1, 4): 5})

    def test_initialize_workplace__yes_bias_10_nns(self):
        workplace = Workplace(n_input=3, n_output=1, bias=1, n_nn=10)
        worker = Worker(workplace)

        worker.initialize_workplace()

        self.assertEqual(len(workplace.nns), 10)
        self.assertEqual(workplace.innov_history, {(0, 4): 0,
                                                   (1, 4): 1,
                                                   (2, 4): 2,
                                                   (3, 4): 3})
        self.assertEqual(workplace.innov_counter, 3)
        self.assertEqual(workplace.node_genes, [0, 1, 1, 1, 2])

        # test if two nns have identical genes except weight
        nn1 = workplace.nns[0]
        nn2 = workplace.nns[9]
        gene1 = nn1.connect_genes
        gene2 = nn2.connect_genes
        gene1_w_removed = np.delete(gene1, 2, 1)
        gene2_w_removed = np.delete(gene2, 2, 1)
        self.assertEqual(gene1.shape, (4, 5))
        self.assertFalse(np.array_equal(gene1, gene2), "two genes must have different weights")
        self.assertTrue(np.array_equal(gene1_w_removed, gene2_w_removed), "two genes have identical other elements")

    def test_initialize_workplace__no_bias_10_nns(self):
        workplace = Workplace(n_input=3, n_output=1, bias=None, n_nn=10)
        worker = Worker(workplace)

        worker.initialize_workplace()

        self.assertEqual(len(workplace.nns), 10)
        self.assertEqual(workplace.innov_history, {(0, 3): 0,
                                                   (1, 3): 1,
                                                   (2, 3): 2})
        self.assertEqual(workplace.innov_counter, 2)
        self.assertEqual(workplace.node_genes, [1, 1, 1, 2])

        # test if two nns have identical genes except weight
        nn1 = workplace.nns[0]
        nn2 = workplace.nns[9]
        gene1 = nn1.connect_genes
        gene2 = nn2.connect_genes
        gene1_w_removed = np.delete(gene1, 2, 1)
        gene2_w_removed = np.delete(gene2, 2, 1)
        self.assertEqual(gene1.shape, (3, 5))
        self.assertFalse(np.array_equal(gene1, gene2), "two genes must have different weights")
        self.assertTrue(np.array_equal(gene1_w_removed, gene2_w_removed), "two genes have identical other elements")

    def test_activate__valid_input(self):
        workplace = Workplace(3, 4)
        worker = Worker(workplace)

        xs = np.array([[1, 2, 3]])
        ws = np.array([[1], [1], [1]])
        y = worker.activate(xs, ws)

        self.assertEqual(y, sigmoid(np.dot(xs, ws)))

    def test_activate__ws_xs_size_mismatch(self):
        workplace = Workplace(3, 4)
        worker = Worker(workplace)

        xs = np.array([[1, 2, 3, 4]])
        ws = np.array([[1], [1], [1]])

        self.assertRaises(AssertionError, worker.activate, xs, ws)

    def test_activate__xs_invalid_shape(self):
        workplace = Workplace(3, 4)
        worker = Worker(workplace)

        xs = np.array([[1], [1], [1]])
        ws = np.array([[1, 2, 3]])

        self.assertRaises(AssertionError, worker.activate, xs, ws)

    def test_get_nodes_in_of_node(self):
        workplace = Workplace(3, 3, bias=1)
        worker = Worker(workplace)
        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 7, 0.5, 1, 0],
                                     [1, 4, 0.5, 1, 1],
                                     [1, 5, 0.5, 1, 2],
                                     [2, 8, 0.5, 1, 3],
                                     [3, 9, 0.5, 1, 4],
                                     [7, 4, 0.5, 1, 5],
                                     [8, 4, 0.5, 1, 6],
                                     [8, 6, 0.5, 1, 7],
                                     [9, 5, 0.5, 1, 8],
                                     [9, 6, 0.5, 1, 9],
                                     [8, 9, 0.5, 1, 10]])

        nodes_in = worker.get_nodes_in_of_node(0, nn)
        self.assertEqual(nodes_in, set())
        nodes_in = worker.get_nodes_in_of_node(1, nn)
        self.assertEqual(nodes_in, set())
        nodes_in = worker.get_nodes_in_of_node(2, nn)
        self.assertEqual(nodes_in, set())
        nodes_in = worker.get_nodes_in_of_node(3, nn)
        self.assertEqual(nodes_in, set())
        nodes_in = worker.get_nodes_in_of_node(4, nn)
        self.assertEqual(nodes_in, {1, 7, 8})
        nodes_in = worker.get_nodes_in_of_node(5, nn)
        self.assertEqual(nodes_in, {1, 9})
        nodes_in = worker.get_nodes_in_of_node(6, nn)
        self.assertEqual(nodes_in, {8, 9})
        nodes_in = worker.get_nodes_in_of_node(7, nn)
        self.assertEqual(nodes_in, {0})
        nodes_in = worker.get_nodes_in_of_node(8, nn)
        self.assertEqual(nodes_in, {2})
        nodes_in = worker.get_nodes_in_of_node(9, nn)
        self.assertEqual(nodes_in, {3, 8})

    def test_get_weight_of_connect(self):
        workplace = Workplace(3, 3, bias=1)
        worker = Worker(workplace)

        nn = NeuralNetwork
        nn.connect_genes = np.array([[0, 7, 1.2, 1, 0],
                                     [1, 4, 3.3, 1, 1],
                                     [1, 5, 1.4, 1, 2],
                                     [2, 8, 4.5, 1, 3],
                                     [3, 9, 1.6, 1, 4],
                                     [7, 4, 2.7, 1, 5],
                                     [8, 4, 5.8, 1, 6],
                                     [8, 6, 1.9, 1, 7],
                                     [9, 5, 0.1, 1, 8],
                                     [9, 6, 0.12, 1, 9],
                                     [8, 9, 0.13, 1, 10]])

        self.assertEqual(worker.get_weight_of_connect(0, 7, nn), 1.2)
        self.assertEqual(worker.get_weight_of_connect(1, 4, nn), 3.3)
        self.assertEqual(worker.get_weight_of_connect(1, 5, nn), 1.4)
        self.assertEqual(worker.get_weight_of_connect(2, 8, nn), 4.5)
        self.assertEqual(worker.get_weight_of_connect(3, 9, nn), 1.6)
        self.assertEqual(worker.get_weight_of_connect(7, 4, nn), 2.7)
        self.assertEqual(worker.get_weight_of_connect(8, 4, nn), 5.8)
        self.assertIsNone(worker.get_weight_of_connect(0, 9, nn))

    def test_calc_output(self):
        workplace = Workplace(3, 3, bias=1, activ_func=linear)
        worker = Worker(workplace)
        inputs = np.array([1, 2, 3])
        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 7, 1, 1, 0],
                                     [1, 4, 3, 1, 1],
                                     [1, 5, 1, 1, 2],
                                     [2, 8, 4, 1, 3],
                                     [3, 9, 1, 1, 4],
                                     [7, 4, 2, 1, 5],
                                     [8, 4, 5, 1, 6],
                                     [8, 6, 1, 1, 7],
                                     [9, 5, 1, 1, 8],
                                     [9, 6, 1, 1, 9],
                                     [8, 9, 1, 1, 10]])

        node_out = 4
        activ_result = [None] * 10
        updated_activ_result = worker.calc_output(node_out, activ_result, inputs, nn)

        self.assertEqual(updated_activ_result, [1, 1, 2, None, 12, None, None, 0.5, 4, None])

    def test_get_output_nodes__yes_bias(self):
        workplace = Workplace(3, 3, bias=1)
        worker = Worker(workplace)

        self.assertEqual(worker.get_output_nodes(), [4, 5, 6])

    def test_get_output_nodes__no_bias(self):
        workplace = Workplace(3, 3)
        worker = Worker(workplace)

        self.assertEqual(worker.get_output_nodes(), [3, 4, 5])

    def test_feedforward__AND(self):
        workplace = Workplace(2, 1, bias=-1)
        workplace.activ_func = linear
        workplace.node_genes = [0, 1, 1, 2]
        worker = Worker(workplace)

        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 3, 2, 1, 0],
                                     [1, 3, 2, 1, 1],
                                     [2, 3, 2, 1, 2]])

        inputs = np.array([0, 0])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [-1])

        inputs = np.array([0, 1])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [0])

        inputs = np.array([1, 0])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [0])

        inputs = np.array([1, 1])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [1])

    def test_feedforward__OR(self):
        workplace = Workplace(2, 1, bias=-1)
        workplace.activ_func = linear
        workplace.node_genes = [0, 1, 1, 2]
        worker = Worker(workplace)

        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 3, 1.0, 1, 0],
                                     [1, 3, 3.0, 1, 1],
                                     [2, 3, 3.0, 1, 2]])

        inputs = np.array([0, 0])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [-0.5])

        inputs = np.array([0, 1])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [1])

        inputs = np.array([1, 0])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [1])

        inputs = np.array([1, 1])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [2.5])

    def test_feedforward__XOR(self):
        workplace = Workplace(2, 1, bias=-1)
        workplace.activ_func = linear
        workplace.node_genes = [0, 1, 1, 2, 3]
        worker = Worker(workplace)

        nn = NeuralNetwork()
        nn.connect_genes = np.array([[0, 3, 0.6, 1, 0],
                                     [1, 3, 2.0, 1, 1],
                                     [2, 3, 2.0, 1, 2],
                                     [1, 4, 2.0, 1, 3],
                                     [4, 3, 2.0, 1, 4],
                                     [2, 4, 2.0, 1, 5],
                                     [0, 4, 2.0, 1, 6]])

        inputs = np.array([0, 0])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [-1.3])

        inputs = np.array([0, 1])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [0.7])

        inputs = np.array([1, 0])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [0.7])

        inputs = np.array([1, 1])
        result = worker.feedforward(inputs, nn)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [2.7])

    def test_get_node_between(self):
        workplace = Workplace(3, 3, bias=1, activ_func=linear)
        worker = Worker(workplace)
        workplace.innov_history = {(0, 7): 0,
                                   (1, 4): 1,
                                   (1, 5): 2,
                                   (2, 8): 3,
                                   (3, 9): 4,
                                   (7, 4): 5,
                                   (8, 4): 6,
                                   (8, 6): 7,
                                   (9, 5): 8,
                                   (9, 6): 9,
                                   (8, 9): 10,
                                   (2, 7): 11,
                                   (7, 6): 12}

        node_in = 2
        node_out = 6
        result = worker.get_node_between(node_in, node_out)
        self.assertTrue(result in (7, 8))

        node_in = 0
        node_out = 4
        result = worker.get_node_between(node_in, node_out)
        self.assertTrue(result in (7, ))

        node_in = 8
        node_out = 6
        result = worker.get_node_between(node_in, node_out)
        self.assertTrue(result in (9, ))

        node_in = 9
        node_out = 6
        result = worker.get_node_between(node_in, node_out)
        self.assertIsNone(result)

    def test_add_node__yes_bias(self):
        workplace = Workplace(2, 1, bias=-1, n_nn=10)
        worker = Worker(workplace)
        worker.initialize_workplace()

        counter = workplace.innov_counter
        history = workplace.innov_history
        node_genes = workplace.node_genes
        self.assertEqual(counter, 2)
        self.assertEqual(history, {(0, 3): 0,
                                   (1, 3): 1,
                                   (2, 3): 2})
        self.assertEqual(node_genes, [0, 1, 1, 2])

        nn = workplace.nns[0]
        node_in = 1
        node_out = 3
        ori_weight = worker.get_weight_of_connect(node_in, node_out, nn)
        worker.add_node(node_in, node_out, nn)

        counter = workplace.innov_counter
        history = workplace.innov_history
        node_genes = workplace.node_genes
        self.assertEqual(counter, 4)
        self.assertEqual(history, {(0, 3): 0,
                                   (1, 3): 1,
                                   (2, 3): 2,
                                   (1, 4): 3,
                                   (4, 3): 4})
        self.assertEqual(node_genes, [0, 1, 1, 2, 3])

        front_connect_weight = worker.get_weight_of_connect(node_in, 4, nn)
        back_connect_weight = worker.get_weight_of_connect(4, node_out, nn)
        self.assertEqual(front_connect_weight, 1)
        self.assertEqual(back_connect_weight, ori_weight)

        self.assertEqual(nn.connect_genes.shape, (5, 5))
        gene_w_removed = np.delete(nn.connect_genes, 2, 1)
        self.assertTrue(np.array_equal(gene_w_removed, np.array([[0, 3, 1, 0],
                                                                 [1, 3, 1, 1],
                                                                 [2, 3, 1, 2],
                                                                 [1, 4, 1, 3],
                                                                 [4, 3, 1, 4]])))

        nn2 = workplace.nns[1]
        node_in = 1
        node_out = 3
        ori_weight = worker.get_weight_of_connect(node_in, node_out, nn2)
        worker.add_node(node_in, node_out, nn2)

        counter = workplace.innov_counter
        history = workplace.innov_history
        node_genes = workplace.node_genes
        self.assertEqual(counter, 4)
        self.assertEqual(history, {(0, 3): 0,
                                   (1, 3): 1,
                                   (2, 3): 2,
                                   (1, 4): 3,
                                   (4, 3): 4})
        self.assertEqual(node_genes, [0, 1, 1, 2, 3])

        front_connect_weight = worker.get_weight_of_connect(node_in, 4, nn2)
        back_connect_weight = worker.get_weight_of_connect(4, node_out, nn2)
        self.assertEqual(front_connect_weight, 1)
        self.assertEqual(back_connect_weight, ori_weight)
        self.assertEqual(nn2.connect_genes.shape, (5, 5))
        gene_w_removed = np.delete(nn2.connect_genes, 2, 1)
        self.assertTrue(np.array_equal(gene_w_removed, np.array([[0, 3, 1, 0],
                                                                 [1, 3, 1, 1],
                                                                 [2, 3, 1, 2],
                                                                 [1, 4, 1, 3],
                                                                 [4, 3, 1, 4]])))

    def test_mutate_connection(self):
        pass

    def test_mutate_weight(self):
        pass

    def test_crossover(self):
        pass
