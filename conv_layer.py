import numpy as np
from scipy.signal import fftconvolve
from skimage.measure import block_reduce

"""
TODO: More numpyfication
"""

class ConvLayer():

    def __init__(self, n_filters, kernel_size, activation='linear', input_shape=(28, 28, 1), padding='same'):
        self.a_in = None

        # Initialize filters
        self.filters = self.initialize_filter(f_size=[n_filters, kernel_size[0], kernel_size[1]])
        self.biases = np.randomn(n_filters)

        self.outputs = []
        self.activation = activation

    def forward_propagation(self, image):
        self.outputs = []
        for filter_nr, filter in enumerate(self.filters):
            self.outputs.append(self.convolution2D(image, filter, self.biases[filter_nr]))

        return self.outputs

    def convolution2D(self, image, filter, bias, stride=1):
        '''
        Convolution of 'filter' over 'image' using stride length 'stride'
        Param1: image, given as 2D np.array
        Param2: filter, given as 2D np.array
        Param3: bias, given as float
        Param4: stride length, given as integer
        Return: 2D array of convolved image
        '''
        h_filter, _ = filter.shape  # get the filter dimensions
        in_dim, _ = image.shape  # image dimensions (NB image must be [NxN])

        out_dim = int(((in_dim - h_filter) / stride) + 1)  # output dimensions
        out = np.zeros((out_dim, out_dim))  # create the matrix to hold the values of the convolution operation

        # convolve each filter over the image
        # Start at y=0
        curr_y = out_y = 0
        # move filter vertically across the image
        while curr_y + h_filter <= in_dim:
            curr_x = out_x = 0
            # move filter horizontally across the image
            while curr_x + h_filter <= in_dim:
                # perform the convolution operation and add the bias
                out[out_y, out_x] = np.sum(filter * image[curr_y:curr_y + h_filter, curr_x:curr_x + h_filter]) + bias
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
        return out


    def back_propagation(self):
        pass


    def softmax(raw_preds):
        '''
        pass raw predictions through softmax activation function
        Param1: raw_preds - np.array
        Return: np.array
        '''
        out = np.exp(raw_preds)  # exponentiate vector of raw predictions
        return out / np.sum(out)  # divide the exponentiated vector by its sum. All values in the output sum to 1.

    def initialize_filter(self, f_size, scale=1.0):
        stddev = scale / np.sqrt(np.prod(f_size))
        return np.random.normal(loc=0, scale=stddev, size=f_size)

    def forward_propagation(self, a_in):
        for filter in

    def back_propagation(self, error):
        """
        layer L
        :param error: error from layer L+1
        F = filter
        d_F = derivative with regards to filter
        d_X = error in input, input for backpropagation in layer L-1
        """
        # last filter
        d_F = convolve(self.inputs[-1], error)

        for filter_nr in reversed(range(len(self.filters)-1)):
            # Update weights
            self.filters[filter_nr] -= d_F
            d_F = convolve(self.inputs[filter_nr])

        # d_F error input to next layer
        return d_F

def convolve(image, filter):
    return fftconvolve(image, filter, mode='same')

def max_pooling(image, stride=2):
    return block_reduce(image, (stride, stride), np.max)

def average_pooling(image, stride=2):
    return block_reduce(image, (stride, stride), np.average())

def initialize_filter(f_size, scale=1.0):
    stddev = scale / np.sqrt(np.prod(f_size))
    return np.random.normal(loc=0, scale=stddev, size=f_size)

class MaxPoolLayer():
    def __init__(self, a_in, stride):
        self.a_in = a_in
        self.stride = stride
        self.a_out = None

    def feed_forward(self, a_in):
        self.a_in = a_in
        self.a_out = block_reduce(self.a_in, (self.stride, self.stride), np.max)
        return self.a_out

    def back_propagation(self):
        pass



