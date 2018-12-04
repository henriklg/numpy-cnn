'''
source: https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
'''

class Convolution:
    '''
    Convolution class of Convolutional Neural Network

    Todo:
        - add arguments to init
        - swap convolution functions (bias etc)
        - functionality for 2D and 3D convolution

    # Arguments
        filters:
        kernel_size:
        stride:
        padding?:
    '''
    def __init__(self, activations_prev_layer,
                weights,
                grad,
                cache
                ):
        this.X = activations_prev_layer
        this.W = weights
        this.dH = grad
        this.cache = cache


    def conv_forward(X, W):
        '''
        The forward computation for a convolution function

        Arguments:
        X -- output activations of the previous layer, numpy array of shape (n_H_prev, n_W_prev) assuming input channels = 1
        W -- Weights, numpy array of size (f, f) assuming number of filters = 1

        Returns:
        H -- conv output, numpy array of size (n_H, n_W)
        cache -- cache of values needed for conv_backward() function
        '''

        # Retrieving dimensions from X's shape
        (n_H_prev, n_W_prev) = X.shape

        # Retrieving dimensions from W's shape
        (f, f) = W.shape

        # Compute the output dimensions assuming no padding and stride = 1
        n_H = n_H_prev - f + 1
        n_W = n_W_prev - f + 1

        # Initialize the output H with zeros
        H = np.zeros((n_H, n_W))

        # Looping over vertical(h) and horizontal(w) axis of output volume
        for h in range(n_H):
            for w in range(n_W):
                x_slice = X[h:h+f, w:w+f]
                H[h,w] = np.sum(x_slice * W)

        # Saving information in 'cache' for backprop
        cache = (X, W)

        return H, cache



    def conv_backward(dH, cache):
        '''
        The backward computation for a convolution function

        Arguments:
        dH -- gradient of the cost with respect to output of the conv layer (H), numpy array of shape (n_H, n_W) assuming channels = 1
        cache -- cache of values needed for the conv_backward(), output of conv_forward()

        Returns:
        dX -- gradient of the cost with respect to input of the conv layer (X), numpy array of shape (n_H_prev, n_W_prev) assuming channels = 1
        dW -- gradient of the cost with respect to the weights of the conv layer (W), numpy array of shape (f,f) assuming single filter
        '''

        # Retrieving information from the "cache"
        (X, W) = cache

        # Retrieving dimensions from X's shape
        (n_H_prev, n_W_prev) = X.shape

        # Retrieving dimensions from W's shape
        (f, f) = W.shape

        # Retrieving dimensions from dH's shape
        (n_H, n_W) = dH.shape

        # Initializing dX, dW with the correct shapes
        dX = np.zeros(X.shape)
        dW = np.zeros(W.shape)

        # Looping over vertical(h) and horizontal(w) axis of the output
        for h in range(n_H):
            for w in range(n_W):
                dX[h:h+f, w:w+f] += W * dH(h,w)
                dW += X[h:h+f, w:w+f] * dH(h,w)
        return dX, dW



def main():
    pass

if __name__ == '__main__':
    main()
