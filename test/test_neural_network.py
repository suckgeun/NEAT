import unittest
import numpy as np
from players.neuralnet import NeuralNetwork

"""
Tests individual neural network(nn) objects
"""


class NeuralNetworkTest(unittest.TestCase):

    def test_create_nn__three_inputs_one_outputs(self):
        nn = NeuralNetwork(3, 1)
        # 0: input
        # 1: output
        # 2: hidden
        self.assertTrue(np.array_equal(nn.node_genes, np.array([0, 0, 0, 1])), "initial nodes")
        self.assertIsNone(nn.connect_genes, "before initialization")







    def test_add_hidden_node(self):
        nn = NeuralNetwork(2, 1)
        self.assertTrue(np.array_equal(nn.node_genes, np.array([0, 0, 1])))

        # connection_gene
        # in, out, weight, enabled, innov
        #  0,   2,     w1,       1,     0, (connection_num = 0)
        #  1,   2,     w2,       1,     1, (connection_num = 1)

        # if connection number is given, add a hidden node there
        # if not, add a hidden node to a random connection.
        nn.add_hidden_node(target_connection=0)
        self.assertTrue(np.array_equal(nn.node_genes, np.array([0, 0, 1, 2])))
        # in, out, weight, enabled, innov
        #  0,   2,     w1,       0,     0, (connection_num = 0)
        #  1,   2,     w2,       1,     1, (connection_num = 1)
        #  0,   3,      1,       1,     2, (connection_num = 2)
        #  3,   2,     w1,       1,     3, (connection_num = 3)
        self.assertEqual(nn.connect_genes[0, 3], 0, "previous connection disabled")

        self.assertEqual(nn.connect_genes[2, 0], 0, "new connection input")
        self.assertEqual(nn.connect_genes[2, 1], 3, "new connection out to hidden")
        self.assertEqual(nn.connect_genes[2, 2], 1, "new connection front weight always one")
        self.assertEqual(nn.connect_genes[2, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[2, 4], 2, "Innovation number")

        self.assertEqual(nn.connect_genes[3, 0], 3, "new connection hidden as input")
        self.assertEqual(nn.connect_genes[3, 1], 2, "new connection output")
        self.assertEqual(nn.connect_genes[3, 2], nn.connect_genes[0, 2], "assign previous weight")
        self.assertEqual(nn.connect_genes[3, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[3, 4], 3, "Innovation number")

        # assign new value to weights
        nn.connect_genes[2, 3] = 0.99
        nn.connect_genes[3, 3] = -0.99
        # in, out, weight, enabled, innov
        #  0,   2,     w1,       0,     0, (connection_num = 0)
        #  1,   2,     w2,       1,     1, (connection_num = 1)
        #  0,   3,   0.99,       1,     2, (connection_num = 2)
        #  3,   2,  -0.99,       1,     3, (connection_num = 3)
        nn.add_hidden_node(target_connection=2)
        self.assertTrue(np.array_equal(nn.node_genes, np.array([0, 0, 1, 2, 2])))
        # in, out, weight, enabled, innov
        #  0,   2,     w1,       0,     0, (connection_num = 0)
        #  1,   2,     w2,       1,     1, (connection_num = 1)
        #  0,   3,   0.99,       0,     2, (connection_num = 2)
        #  3,   2,  -0.99,       1,     3, (connection_num = 3)
        #  0,   4,      1,       1,     4, (connection_num = 4)
        #  4,   3,   0.99,       1,     5, (connection_num = 5)
        self.assertEqual(nn.connect_genes[2, 3], 0, "previous connection disabled")

        self.assertEqual(nn.connect_genes[4, 0], 0, "new connection input")
        self.assertEqual(nn.connect_genes[4, 1], 4, "new connection out to hidden")
        self.assertEqual(nn.connect_genes[4, 2], 1, "new connection front weight always one")
        self.assertEqual(nn.connect_genes[4, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[4, 4], 4, "Innovation number")

        self.assertEqual(nn.connect_genes[5, 0], 4, "new connection hidden as input")
        self.assertEqual(nn.connect_genes[5, 1], 3, "new connection output")
        self.assertEqual(nn.connect_genes[5, 2], nn.connect_genes[2, 2], "assign previous weight")
        self.assertEqual(nn.connect_genes[5, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[5, 4], 5, "Innovation number")


def main():
    unittest.main()


if __name__ == '__main__':
    main()
