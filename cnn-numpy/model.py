import numpy as np
from conv_layer import ConvLayer, MaxPoolLayer, FullyConnectedLayer
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class CNN():

    def __init__(self, X_data, Y_data):
        self.layers = []
        self.X = X_data
        self.Y = Y_data

        self.n_categories = np.shape(self.Y)[1]
        self.n_images = np.shape(self.Y)[0]

    # Functions to add various layers to the model
    def add_conv_layer(self, n_filters, kernel_size, eta=0.001):
        self.layers.append(ConvLayer(n_filters, kernel_size, eta))

    def add_maxpool_layer(self, stride=2):
        self.layers.append(MaxPoolLayer(stride))

    def add_fullyconnected_layer(self, eta=0.1):
        self.layers.append(FullyConnectedLayer(self.n_categories, self.n_images, eta))

    # Define data for training or testing
    def new_input(self, X_data, Y_data):
        self.X = X_data
        self.Y = Y_data
        self.n_images = np.shape(self.Y)[0]
        self.layers[-1].update_images(self.n_images)

    # Forward propagation through all layers
    def forward_propagation(self):
        input = self.X
        for layer in self.layers:
            new_input = layer.feed_forward(input)
            input = new_input

        return new_input

    # Back propagation through all layers in revers order
    def back_propagation(self, y_hat):
        # Derivative of loss function wrt softmax
        error = y_hat - self.Y

        # propagate error through layers
        for layer in reversed(self.layers):
            new_error = layer.back_propagation(error)
            error = new_error

    # Loss function for multiclass classifier
    def loss(self, y, target):
        return -np.sum(target*np.log(y), axis=1)

    # Predict class from model output
    def predict(self, y_hat):
        out = np.zeros(shape=np.shape(y_hat))

        # Predict by setting largest propbability to 1
        out[np.arange(np.shape(y_hat)[0]), np.argmax(y_hat, axis=1).T] = 1
        return out

    # Accuracy
    def accuracy(self, y_hat):
        y_ = self.predict(y_hat)
        a = np.sum(np.all(y_ == self.Y, axis=1))
        print('accuracy: ', a / np.shape(y_)[0])

    # display confusion matrix
    def confusion_matrix(self, y_hat, target):
        # Calculate confusion matrix
        cnf_matrix = confusion_matrix(target, y_hat)

        # Print confusion matrix
        np.set_printoptions(precision=2)
        print(cnf_matrix)

        # Plot confusion matrix
        classes = np.arange(0, 10)
        plt.imshow(cnf_matrix, interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

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

def transform_targets_back(targets):
    return np.where(targets == 1)[1]

if __name__ == '__main__':
    np.random.seed(100)
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))

    # Set up model
    X_train, X_test, Y_train, Y_test = train_test_split(digits.images, transform_targets(digits.target), test_size=0.2)
    model = CNN(X_train[0:100, :], Y_train[0:100, :])
    model.add_conv_layer(n_filters=5, kernel_size=[2, 2])
    model.add_maxpool_layer(stride=2)
    model.add_conv_layer(n_filters=2, kernel_size=[2, 2])
    model.add_maxpool_layer(stride=2)
    model.add_fullyconnected_layer()

    # Train model with batch gradient descent
    n_images = np.shape(X_train)[0]
    indices = np.arange(0, n_images)
    batch = 10

    for i in range(20):
        for j in range(int(n_images/batch)):
            r_indices = np.random.choice(indices, size=batch)
            model.new_input(X_train[r_indices, :], Y_train[r_indices, :])
            pred = model.forward_propagation()
            model.back_propagation(pred)

        print('Total loss', np.sum(model.loss(pred, Y_train[r_indices, :])))
        print('finished round ', i)

    # Training data accuracy
    model.new_input(X_train, Y_train)
    predict_train = model.forward_propagation()
    print('Training data accuracy')
    model.accuracy(Y_train)

    # Test data accuracy
    model.new_input(X_test, Y_test)
    predict_test = model.forward_propagation()
    print('Test data accuracy')
    model.accuracy(predict_test)

    # Confusion matrix
    pred = transform_targets_back(model.predict(predict_test))
    tar = transform_targets_back(Y_test)
    print(np.shape(pred), np.shape(tar))
    model.confusion_matrix(pred, tar)



