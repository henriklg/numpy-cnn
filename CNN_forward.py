'''
Convolutional neural network - forward propegation

TODO:
- Make 3D and 2D into one function
- Run on GPU?
'''

import numpy as np

def convolution3D(image, filter, bias, stride=1):
    '''
    Convolution of 'filter' over 'image' using stride length 'stride'

    Param1: image, given as 3D np.array
    Param2: filter, given as 3D np.array
    Param3: bias, given as list
    Param4: stride length, given as integer

    Return: 3D array of convolved image
    '''
    (n_f, n_c_f, f, _) = filter.shape   # get the filter dimensions
    n_c, in_dim, _ = image.shape        # image dimensions
    out_dim = int(((in_dim - f)/stride)+1)    # output dimensions

    # ensure that the filter dimensions match the dimensions of the input image
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"

    out = np.zeros((n_f,out_dim,out_dim)) # create the matrix to hold the values of the convolution operation

    # convolve each filter over the image
    for curr_f in range(n_f):
        curr_y = out_y = 0
        # move filter vertically across the image
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            # move filter horizontally across the image
            while curr_x + f <= in_dim:
                # perform the convolution operation and add the bias
                out[curr_f, out_y, out_x] = np.sum(filter[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
    return out


def convolution2D(image, filter, bias, stride=1):
    '''
    Convolution of 'filter' over 'image' using stride length 'stride'

    Param1: image, given as 2D np.array
    Param2: filter, given as 2D np.array
    Param3: bias, given as float
    Param4: stride length, given as integer

    Return: 2D array of convolved image
    '''
    f, _ = filter.shape   # get the filter dimensions
    in_dim, _ = image.shape        # image dimensions

    out_dim = int(((in_dim - f)/stride)+1)    # output dimensions
    out = np.zeros((out_dim,out_dim)) # create the matrix to hold the values of the convolution operation

    # convolve each filter over the image
    curr_y = out_y = 0
    # move filter vertically across the image
    while curr_y + f <= in_dim:
        curr_x = out_x = 0
        # move filter horizontally across the image
        while curr_x + f <= in_dim:
            # perform the convolution operation and add the bias
            out[out_y, out_x] = np.sum(filter * image[curr_y:curr_y+f, curr_x:curr_x+f]) + bias
            curr_x += stride
            out_x += 1
        curr_y += stride
        out_y += 1
    return out


def maxpool3D(image, kernel_size=2, step_size=2):
    '''
    Downsample input 'image' using a kernel size of 'kernel_size' and a stride of 'step_size'

    Param1: image given - 3D np.array
    Param2: kernel_size - integer
    Param3: step_size - integer

    Return: downsampled images - 3D np.array
    '''
    n_c, h_prev, w_prev = image.shape

    # calculate output dimensions after the maxpooling operation.
    h = int((h_prev - kernel_size)/step_size)+1
    w = int((w_prev - kernel_size)/step_size)+1

    # create a matrix to hold the values of the maxpooling operation.
    downsampled = np.zeros((n_c,h,w))
    # slide the window over every part of the image using stride step_size. Take the maximum value at each step.
    for i in range(n_c):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + kernel_size <= h_prev:
            curr_x = out_x = 0
            # slide the max pooling window horizontally across the image
            while curr_x + kernel_size <= w_prev:
                # choose the maximum value within the window at each step and store it to the output matrix
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+kernel_size, curr_x:curr_x+kernel_size])
                curr_x += step_size
                out_x += 1
            curr_y += step_size
            out_y += 1
    return downsampled


def maxpool2D(image, kernel_size=2, step_size=2):
    '''
    Downsample input 'image' using a kernel size of 'kernel_size' and a stride of 'step_size'

    Param1: image given - 2D np.array
    Param2: kernel_size - integer
    Param3: step_size - integer

    Return: downsampled images - 2D np.array
    '''
    h_in, w_in = image.shape

    # calculate output dimensions after the maxpooling operation.
    h_out = int((h_in - kernel_size)/step_size)+1
    w_out = int((w_in - kernel_size)/step_size)+1

    # create a matrix to hold the values of the maxpooling operation.
    downsampled = np.zeros((h_out,w_out))
    # slide the window over every part of the image using stride step_size. Take the maximum value at each step.

    curr_y = out_y = 0
    # slide the max pooling window vertically across the image
    while curr_y + kernel_size <= h_in:
        curr_x = out_x = 0
        # slide the max pooling window horizontally across the image
        while curr_x + kernel_size <= w_in:
            # choose the maximum value within the window at each step and store it to the output matrix
            downsampled[out_y, out_x] = np.max(image[curr_y:curr_y+kernel_size, curr_x:curr_x+kernel_size])
            curr_x += step_size
            out_x += 1
        curr_y += step_size
        out_y += 1
    return downsampled


def softmax(raw_preds):
    '''
    pass raw predictions through softmax activation function
    '''
    out = np.exp(raw_preds) # exponentiate vector of raw predictions
    return out/np.sum(out) # divide the exponentiated vector by its sum. All values in the output sum to 1.
