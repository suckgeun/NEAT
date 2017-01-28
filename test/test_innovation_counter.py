import unittest
from globaladmin import nngen

"""
Tests global incrementation of Innovation count
"""


class InnovationCounterTest(unittest.TestCase):

    def test_create_one_nn__three_inputs_one_outputs(self):
        nns = nngen.create_nns(3, 1, 1)
        self.assertEqual(len(nns), 1, "created one object")

        # check global innovation count
        self.assertEqual(nngen.INNOVATION_COUNTER, 2)

    def test_create_two_nn__two_inputs_three_outputs(self):
        nns = nngen.create_nns(2, 3, 2)
        self.assertEqual(len(nns), 2, "created two objects")

        # check global innovation count
        self.assertEqual(nngen.INNOVATION_COUNTER, 5)

    def test_create_hundred_nn__hundred_inputs_ten_outputs(self):
        nns = nngen.create_nns(100, 10, 100)
        self.assertEqual(len(nns), 100, "created hundred objects")

        # check global innovation count
        self.assertEqual(nngen.INNOVATION_COUNTER, 999)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
