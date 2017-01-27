import unittest
import numpy as np

"""
Tests individual neural network(nn) objects
"""


class NeuralNetworkTest(unittest.TestCase):

    def create_one_nn__three_inputs_one_outputs(self):
        nns = NnGenerator(3, 1, 1)
        self.assertEqual(len(nns), 1, "created one object")
        nn = nns[0]
        # check type
        self.assertIsInstance(nn, NeuralNetwork, "object type")
        # 0: input
        # 1: output
        # 2: hidden
        self.assertEqual(nn.node_genes, np.array([0, 0, 0, 1]))
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

    def create_one_nn__two_inputs_three_outputs(self):
        nns = NnGenerator(2, 3, 1)
        self.assertEqual(len(nns), 1, "created one object")
        nn = nns[0]
        # check type
        self.assertIsInstance(nn, NeuralNetwork, "object created")
        # 0: input
        # 1: output
        # 2: hidden
        self.assertEqual(nn.node_genes, np.array([0, 0, 1, 1, 1]))
        # in, out, weight, enabled, innov
        #  0,   2,     w1,       1,     0,
        #  1,   2,     w2,       1,     1,
        #  0,   3,     w3,       1,     2,
        #  1,   3,     w4,       1,     3,
        #  0,   4,     w5,       1,     4,
        #  1,   4,     w6,       1,     5,
        self.assertEqual(nn.connect_genes.shape, (6, 5), "shape of connect genes")

        self.assertEqual(nn.connect_genes[0, 0], 0, "input")
        self.assertEqual(nn.connect_genes[0, 1], 2, "output")
        self.assertIsInstance(nn.connect_genes[0, 2], float, "weight")
        self.assertEqual(nn.connect_genes[0, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[0, 4], 0, "innovation number")

        self.assertEqual(nn.connect_genes[1, 0], 1, "input")
        self.assertEqual(nn.connect_genes[1, 1], 2, "output")
        self.assertIsInstance(nn.connect_genes[1, 2], float, "weight")
        self.assertEqual(nn.connect_genes[1, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[1, 4], 1, "innovation number")

        self.assertEqual(nn.connect_genes[2, 0], 0, "input")
        self.assertEqual(nn.connect_genes[2, 1], 3, "output")
        self.assertIsInstance(nn.connect_genes[2, 2], float, "weight")
        self.assertEqual(nn.connect_genes[2, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[2, 4], 2, "innovation number")

        self.assertEqual(nn.connect_genes[3, 0], 1, "input")
        self.assertEqual(nn.connect_genes[3, 1], 3, "output")
        self.assertIsInstance(nn.connect_genes[3, 2], float, "weight")
        self.assertEqual(nn.connect_genes[3, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[3, 4], 3, "innovation number")

        self.assertEqual(nn.connect_genes[4, 0], 0, "input")
        self.assertEqual(nn.connect_genes[4, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[4, 2], float, "weight")
        self.assertEqual(nn.connect_genes[4, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[4, 4], 4, "innovation number")

        self.assertEqual(nn.connect_genes[5, 0], 1, "input")
        self.assertEqual(nn.connect_genes[5, 1], 4, "output")
        self.assertIsInstance(nn.connect_genes[5, 2], float, "weight")
        self.assertEqual(nn.connect_genes[5, 3], 1, "enabled")
        self.assertEqual(nn.connect_genes[5, 4], 5, "innovation number")

    def create_two_nn__two_inputs_one_outputs(self):
        nns = NnGenerator(2, 1, 2)
        self.assertEqual(len(nns), 2, "created one object")
        nn0 = nns[0]
        nn1 = nns[1]
        # check type
        self.assertIsInstance(nn0, NeuralNetwork, "object created")
        self.assertIsInstance(nn1, NeuralNetwork, "object created")
        # 0: input
        # 1: output
        # 2: hidden
        self.assertEqual(nn0.node_genes, np.array([0, 0, 1]))
        self.assertEqual(nn1.node_genes, np.array([0, 0, 1]))
        # in, out, weight, enabled, innov
        #  0,   2,     w1,       1,     0,
        #  1,   2,     w2,       1,     1,
        self.assertEqual(nn0.connect_genes.shape, (2, 5), "shape of connect genes")

        self.assertEqual(nn0.connect_genes[0, 0], 0, "input")
        self.assertEqual(nn0.connect_genes[0, 1], 2, "output")
        self.assertIsInstance(nn0.connect_genes[0, 2], float, "weight")
        self.assertEqual(nn0.connect_genes[0, 3], 1, "enabled")
        self.assertEqual(nn0.connect_genes[0, 4], 0, "innovation number")

        self.assertEqual(nn0.connect_genes[1, 0], 1, "input")
        self.assertEqual(nn0.connect_genes[1, 1], 2, "output")
        self.assertIsInstance(nn0.connect_genes[1, 2], float, "weight")
        self.assertEqual(nn0.connect_genes[1, 3], 1, "enabled")
        self.assertEqual(nn0.connect_genes[1, 4], 1, "innovation number")

        # in, out, weight, enabled, innov
        #  0,   2,     w1,       1,     0,
        #  1,   2,     w2,       1,     1,
        self.assertEqual(nn1.connect_genes.shape, (2, 5), "shape of connect genes")

        self.assertEqual(nn1.connect_genes[0, 0], 0, "input")
        self.assertEqual(nn1.connect_genes[0, 1], 2, "output")
        self.assertIsInstance(nn1.connect_genes[0, 2], float, "weight")
        self.assertEqual(nn1.connect_genes[0, 3], 1, "enabled")
        self.assertEqual(nn1.connect_genes[0, 4], 0, "innovation number")

        self.assertEqual(nn1.connect_genes[1, 0], 1, "input")
        self.assertEqual(nn1.connect_genes[1, 1], 2, "output")
        self.assertIsInstance(nn1.connect_genes[1, 2], float, "weight")
        self.assertEqual(nn1.connect_genes[1, 3], 1, "enabled")
        self.assertEqual(nn1.connect_genes[1, 4], 1, "innovation number")


def main():
    unittest.main()


if __name__ == '__main__':
    main()
