import unittest
import numpy as np
from CNN_forward import convolution2D, maxpool2D, softmax

# py -m unittest TestCNN.py

class TestCNN(unittest.TestCase):
    '''
    Basic test class for CNN.py
    '''

    def test_convolution2D(self):
        '''
        Test the convolution function of CNN
        '''
        image = np.array([[1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0]])
        filter = np.array([[1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]])
        expected = np.array([[4, 3, 4],
                    [2, 4, 3],
                    [2, 3, 4]])
        bias = 0
        stride = 1

        image_after_convolution = convolution2D(image, filter, bias, stride)
        self.assertListEqual(image_after_convolution.tolist(),expected.tolist())


    def test_maxpool2D(self):
        image = np.array([[1, 0, 2, 3],
                [4, 6, 6, 8],
                [3, 1, 1, 0],
                [1, 2, 2, 4]])
        expected = np.array([[6, 8],
                            [3,4]])
        image_reduced = maxpool2D(image)
        self.assertListEqual(expected.tolist(), image_reduced.tolist())


    def test_softmax(self):
        predictions = np.zeros(4)
        result = softmax(predictions)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        self.assertListEqual(expected.tolist(), result.tolist())


    if __name__ == '__main__':
        unittest.main()
