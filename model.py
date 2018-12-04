import numpy as np
from conv_layer import ConvLayer, MaxPoolLayer, FullyConnectedLayer
from sklearn import datasets
from sklearn.model_selection import train_test_split

class CNN():

    def __init__(self, X_data, Y_data):
        self.layers = []
        self.X = X_data
        self.Y = Y_data

        self.n_categories = np.shape(self.Y)[1]
        self.n_images = np.shape(self.Y)[0]

    def add_conv_layer(self, n_filters, kernel_size, eta=0.001):
        self.layers.append(ConvLayer(n_filters, kernel_size, eta))

    def add_maxpool_layer(self, stride=2):
        self.layers.append(MaxPoolLayer(stride))

    def add_fullyconnected_layer(self, eta=0.1):
        self.layers.append(FullyConnectedLayer(self.n_categories, self.n_images, eta))

    def new_input(self, X_data, Y_data):
        self.X = X_data
        self.Y = Y_data
        self.n_images = np.shape(self.Y)[0]
        self.layers[-1].update_images = self.n_images

    def forward_propagation(self):
        input = self.X
        for layer in self.layers:
            new_input = layer.feed_forward(input)
            input = new_input

        return new_input

    def back_propagation(self, y_hat):
        error = y_hat - self.Y
        #print('Error: ', np.sum(error))
        print(y_hat[0, :])
        print(self.loss(y_hat[0, :], self.Y[0, :]))
        print(self.predict(y_hat)[0, :])

        for layer in reversed(self.layers):
            new_error = layer.back_propagation(error)
            error = new_error

    def loss(self, y, target):
        return -np.sum(target*np.log(y))

    def predict(self, y_hat):
        out = np.zeros(shape=np.shape(y_hat))
        out[:, np.argmax(out, axis=1)] = 1
        return out

def transform_targets(targets):
    """transform targets from a n array with values 0-9 to a nx10 array where
    each row is zero, except at the indice corresponding to the value in the
    original array"""
    n = len(targets)
    new_targets = np.zeros([n, 10])
    for i in range(n):
        value = int(targets[i])
        new_targets[i, value] = 1.0
    return new_targets

if __name__ == '__main__':
    np.random.seed(100)
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))
    X_train, X_test, Y_train, Y_test = train_test_split(digits.images, transform_targets(digits.target), test_size=0.9)
    model = CNN(X_train[0:100, :], Y_train[0:100, :])
    model.add_conv_layer(n_filters=10, kernel_size=[2, 2])
    model.add_maxpool_layer(stride=2)
    model.add_conv_layer(n_filters=2, kernel_size=[2, 2])
    model.add_maxpool_layer(stride=2)
    model.add_fullyconnected_layer()

    print(Y_train[0, :])
    for i in range(100):
        pred = model.forward_propagation()
        model.back_propagation(pred)
        print('finished round ', i)


