import unittest
from players.activation import sigmoid


class ActivationTest(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)

