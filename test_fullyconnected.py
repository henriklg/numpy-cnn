from conv_layer import FullyConnectedLayer
import unittest
import numpy as np

class Test(unittest.TestCase):
    # def test_vectorize(self):
    #     FC = FullyConnectedLayer(n_categories=)
    #
    #     a_in = np.ones(shape = [10, 2, 2])
    #     self.assertEqual(len(FC.feed_forward(a_in)), 10*2*2)
    #
    # def test_back_propagation(self):
    #     FC = FullyConnectedLayer(n_categories=5)
    #     a_in = np.ones(shape = [10, 2, 2])
    #     FC.feed_forward(a_in)
    #
    #
    #     y_hat = np.ones(40)
    #     print(FC.back_propagation(y_hat))

    def test_feed_forward(self):
        FC = FullyConnectedLayer(n_categories=5)

        a_in = np.ones(shape=[10, 2, 2])
        y_prob = FC.feed_forward(a_in)
        self.assertEqual(len(y_prob), 5)
        self.assertAlmostEqual(np.sum(y_prob), 1)


if __name__ == '__main__':
    unittest.main()
