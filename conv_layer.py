import numpy as np
from scipy.signal import fftconvolve
from skimage.measure import block_reduce
from math import ceil

"""
TODO: More numpyfication
"""

class ConvLayer():
    def __init__(self, n_filters, kernel_size, activation='linear', input_shape=(28, 28, 1), padding='same'):
        self.a_in = None

        # Initialize filters
        self.filters = self.initialize_filter(f_size=[n_filters, kernel_size[0], kernel_size[1]])
        self.n_filters = n_filters
        self.kernel_size = kernel_size[0]
        self.biases = np.zeros(n_filters)

        self.out = None
        self.z = []
        self.activation = activation

    def forward_propagation(self, images):
        self.a_in = images
        n_images, img_size, _ = np.shape(images)

        stride = 1
        out_img_size = int((img_size-self.kernel_size)/stride + 1)
        self.out = np.zeros(shape=[n_images*self.n_filters, out_img_size, out_img_size])
        self.z = []

        out_nr = 0

        for img_nr in range(n_images):
            image = self.a_in[img_nr, :, :]
            for filter_nr, filter in enumerate(self.filters):
                self.z.append(self.convolution2D(image, filter, self.biases[filter_nr]))
                self.out[out_nr, :, :] = self.convolution2D(image, filter, self.biases[filter_nr])
                out_nr += 1

        return self.out

    def convolution2D(self, image, filter, bias, stride=1, padding=0):
        '''
        Convolution of 'filter' over 'image' using stride length 'stride'
        Param1: image, given as 2D np.array
        Param2: filter, given as 2D np.array
        Param3: bias, given as float
        Param4: stride length, given as integer
        Return: 2D array of convolved image
        '''

        h_filter, _ = filter.shape  # get the filter dimensions

        if padding > 0:
            image = np.pad(image, pad_width=padding, mode='constant')
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

    def softmax(self, raw_preds):
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

    def back_propagation(self, error):
        """
        layer L
        :param error: error from layer L+1
        F = filter
        d_F = derivative with regards to filter
        d_X = error in input, input for backpropagation in layer L-1
        """
        error_out = np.zeros(shape=np.shape(self.a_in))
        n_images, _, _ = np.shape(self.a_in)
        # Backpropagate through activation function part
        for img_nr in range(n_images):
            image = self.a_in[img_nr, :, :]
            for filter_nr in range(len(self.filters)):
                prop_error = error[filter_nr, :, :]*self.d_activation(self.z[filter_nr])
                d_F = self.convolution2D(image, np.rot90(prop_error, 2), bias=0)

                # Update weights
                self.filters[filter_nr] += d_F
                #self.biases[filter_nr] += np.sum()

                error_out[img_nr, :, :] += self.convolution2D(prop_error, np.rot90(self.filters[filter_nr], 2), bias=0, padding=1)

        return error_out

    def activation(self, a):
        a[a <= 0] = 0
        return a

    def d_activation(self, a):
        # Return ReLU derivative
        a[a <= 0] = 0
        a[a > 0] = 1
        return a


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
    def __init__(self, stride):
        self.a_in = None
        self.stride = stride
        self.a_out = None

    def feed_forward(self, a_in):
        self.a_in = a_in
        n_images, img_size, _ = np.shape(self.a_in)
        self.a_out = np.zeros(shape=[n_images, int(ceil(img_size/self.stride)), int(ceil(img_size/self.stride))])
        for img_nr in range(n_images):
            self.a_out[img_nr, :, :] = block_reduce(self.a_in[img_nr, :, :], (self.stride, self.stride), np.max)
        return self.a_out

    def back_propagation(self, d_error):
        # Pass error to pixel with largest value
        e_i = 0
        e_j = 0
        i = 0
        j = 0

        d_p = np.zeros(self.a_in.shape)
        for img_nr in range(np.shape(self.a_in)[0]):
            while i < np.shape(d_p)[1]:
                e_j = 0
                j = 0

                while j < np.shape(d_p)[2]:
                    a = self.a_in[img_nr, i:i+self.stride, j:j+self.stride]
                    x, y  = np.unravel_index(a.argmax(), [self.stride, self.stride])

                    d_p[img_nr, x+i, y+j] = d_error[img_nr, e_i, e_j]
                    e_j += 1
                    j += self.stride

                e_i += 1
                i += self.stride

        return d_p

class FullyConnectedLayer:
    def __init__(self, n_categories, n_images, activation='softmax'):
        # Input
        self.a_in = None

        # Vectorized
        self.S = None

        # Weights
        self.weights = None
        self.bias = np.random.randn(n_categories)
        self.n_categories = n_categories
        self.n_images = n_images
        self.activation = activation
        self.out = None

    def new_shape(self, a):
        # Vectorize step
        lengde, size, size = np.shape(a)
        per_image = int(lengde / self.n_images)

        reshaped = np.zeros(shape=[self.n_images, per_image*size*size])

        count = 0

        for img in range(self.n_images):
            vec = np.zeros(per_image*size*size)
            pos = 0
            while count % per_image != 0:
                vec[pos:pos + size] = (np.ravel(a[count].T))
                count += 1
                pos += size

            reshaped[img, :] = vec
        return reshaped

    def feed_forward(self, a_in):
        # Vectorize step
        self.a_in = a_in
        self.S = self.new_shape(a_in)

        if self.weights is None:
            # initialize weights
            self.weights = np.random.randn(self.n_categories, np.shape(self.S)[1])

        self.out = self.activation_function(np.matmul(self.S, self.weights) + self.bias)

        return self.out

    def activation_function(self, a):
        if self.activation == 'softmax':
            return np.exp(a) / (np.exp(a).sum())

    def back_propagation(self, error):
        # Error w.r.t weights
        w_d = np.matmul(error, self.S.T) #*self.d_activation(self.out) # If categorical loss ????
        b_d = error*self.out #????????????????

        # Calculate error in input
        d_S = np.matmul(self.weights.T, error)#*self.d_activation

        # Reshape
        d_a_in = d_S.reshape(np.shape(self.a_in))

        # Update weights & bias
        self.weights += w_d
        self.bias += b_d

        return d_a_in


