import unittest
from CNN import convolution

class TestCNN(unittest.TestCase):
    '''
    Basic test class for CNN.py
    '''

    def test_convolution(self):
        '''
        Test the convolution function of CNN
        '''
        image = [[1, 1, 1, 0, 0]
                [0, 1, 1, 1, 0]
                [0, 0, 1, 1, 1]
                [0, 0, 1, 1, 0]
                [0, 1, 1, 0, 0]]
        filter = [[1, 0, 1]
                [0, 1, 0]
                [1, 0, 1]]
        expected = [[4, 3, 4]
                    [2, 4, 3]
                    [2, 3, 9]] #234

        bias = 0
        stride = 1

        image_after_convolution = convolution(image, filter, bias, stride)
        print (image_after_convolution)
        print ("\n")
        print (expected)
        self.assertListEqual(image_after_convolution.tolist(),expected.tolist())



    if __name__ == '__main__':
        unittest.main()
