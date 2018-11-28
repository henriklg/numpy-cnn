import unittest
from conv_layer import MaxPoolLayer
import numpy as np

class Test(unittest.TestCase):

    def test_backpropagation(self):
        ML = MaxPoolLayer(stride=2)


if __name__ == '__main__':
    unittest.main()