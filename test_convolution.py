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
        #print(out[0])
        #print(input[0])

        error = np.random.randn(2, 4, 4)
        back = ML.back_propagation(error)

        #print(np.shape(back))

    def test_forward(self):
        input = digits.images[0:2]
        CL = ConvLayer(n_filters=2, kernel_size=[2, 2], activation='ReLU', input_shape=(8, 8, 1))
        ML = MaxPoolLayer(stride=2)

        C_out = CL.forward_propagation(images=input)
        M_out = ML.feed_forward(C_out)
        #print(np.shape(M_out))

    def test_reshape(self):
        a = np.ones(shape=[8, 7, 7])
        F = FullyConnectedLayer(2, 2)
        out = F.new_shape(a)
        #print(np.shape(out))
        _ = F.feed_forward(a)
        out2 = F.reshape_back(out)
        #print(np.shape(out))

    def test_forward_propagation(self):
        input = digits.images[0:2]

        layer1 = ConvLayer(n_filters=2, kernel_size=[2, 2], activation='ReLU', input_shape=(8, 8, 1))
        layer2 = MaxPoolLayer(stride=2)
        layer3 = FullyConnectedLayer(n_categories=10, n_images=2)

        out1 = layer1.forward_propagation(input)
        out2 = layer2.feed_forward(out1)
        print('in: ', out2)
        prob = layer3.feed_forward(out2)
        print(prob)
        self.assertEqual(10, np.shape(prob))





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
