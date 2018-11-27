class Maxpool:
    '''
    Maxpool class of Convolutional Neural Network

    Todo:
        - add arguments to init
        - add functionality for 3D and 2D maxpooling
    '''
    def __init__(self,image,kernel_size,stride):
        self.image = image
        self.kernel_size = kernel_size
        self.stride = stride



    def maxpool2D(image, kernel_size=2, stride=2):
        '''
        Downsample input 'image' using a kernel size of 'kernel_size' and a stride of 'step_size'

        Param1: image given - 2D np.array
        Param2: kernel_size - integer
        Param3: step_size - integer

        Return: downsampled images - 2D np.array
        '''
        h_in, w_in = image.shape

        # calculate output dimensions after the maxpooling operation.
        h_out = int((h_in - kernel_size)/stride)+1
        w_out = int((w_in - kernel_size)/stride)+1

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
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
        return downsampled


    def maxpool2DBackward(dpool, orig, kernel_size, stride):
        '''
        Backpropagation through a maxpooling layer.
        The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
        '''
        orig_dim, _ = orig.shape
        dout = np.zeros(orig.shape)

        curr_y = out_y = 0
        while curr_y + kernel_size <= orig_dim:
            curr_x = out_x = 0
            while curr_x + kernel_size <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_y:curr_y+kernel_size, curr_x:curr_x+kernel_size])
                dout[curr_y+a, curr_x+b] = dpool[out_y, out_x]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
        return dout


if __name__ == "__main__":
    pass
