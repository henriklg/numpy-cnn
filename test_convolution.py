import unittest
import numpy as np
from conv_layer import ConvLayer, MaxPoolLayer, FullyConnectedLayer
from sklearn import datasets

digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))

class Test(unittest.TestCase):

    def test_convolve(self):
        CL = ConvLayer(n_filters=2, kernel_size=[2, 2], activation='ReLU', input_shape=(8, 8, 1))
        #print(np.shape(digits.images[0:2]))
        out = CL.forward_propagation(digits.images[0:2])

        error = np.random.randn(4, 7, 7)
        back = CL.back_propagation(error)

        #print(np.shape(back))

    def test_ML_back(self):
        ML = MaxPoolLayer(stride=2)
        input = digits.images[0:2]

        out = ML.feed_forward(input)
        print(out[0])
        print(input[0])

        error = np.random.randn(2, 4, 4)
        back = ML.back_propagation(error)

        print(np.shape(back))

    def test_forward(self):
        input = digits.images[0:2]
        CL = ConvLayer(n_filters=2, kernel_size=[2, 2], activation='ReLU', input_shape=(8, 8, 1))
        ML = MaxPoolLayer(stride=2)

        C_out = CL.forward_propagation(images=input)
        M_out = ML.feed_forward(C_out)
        print(np.shape(M_out))

    # def test_convolve(self):
    #     image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #     filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    #
    #     right_results = np.array([[-13, -20, -17], [-18, -24, -18], [13, 20, 17]])
    #     self.assertSequenceEqual(convolve(image, filter).tolist(), right_results.tolist())
    #
    # def test_max_pooling(self):
    #     image = np.array([[12, 20, 30, 0], [8, 12, 2, 0], [34, 70, 37, 4], [112, 100, 25, 12]])
    #     right_result = np.array([[20, 30], [112, 37]])
    #
    #     self.assertSequenceEqual(max_pooling(image, stride=2).tolist(), right_result.tolist())
    #
    # def test_average_pooling(self):
    #     pass
    #
    # def test_initialize_filters(self):
    #     fsize = [2, 3, 3]
    #     filters = initialize_filter(f_size=fsize)
    #     self.assertSequenceEqual(np.shape(filters), [2, 3, 3])


        

if __name__ == '__main__':
    unittest.main()
