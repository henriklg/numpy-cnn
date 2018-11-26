import unittest
import numpy as np
from conv_layer import convolve, max_pooling, average_pooling, initialize_filter

class Test(unittest.TestCase):
    def test_convolve(self):
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        right_results = np.array([[-13, -20, -17], [-18, -24, -18], [13, 20, 17]])
        self.assertSequenceEqual(convolve(image, filter).tolist(), right_results.tolist())

    def test_max_pooling(self):
        image = np.array([[12, 20, 30, 0], [8, 12, 2, 0], [34, 70, 37, 4], [112, 100, 25, 12]])
        right_result = np.array([[20, 30], [112, 37]])

        self.assertSequenceEqual(max_pooling(image, stride=2).tolist(), right_result.tolist())

    def test_average_pooling(self):
        pass

    def test_initialize_filters(self):
        fsize = [2, 3, 3]
        filters = initialize_filter(f_size=fsize)
        self.assertSequenceEqual(np.shape(filters), [2, 3, 3])

if __name__ == '__main__':
    unittest.main()
