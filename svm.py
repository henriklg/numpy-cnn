# Import libraries
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata

# mnist = fetch_mldata('MNIST original')
# print(mnist.keys())
#
# images = mnist.data
# targets = mnist.target

digits = datasets.load_digits()

images = digits.images.reshape((len(digits.images), -1))
targets = digits.target

# Show a random image with label
rand_idx = np.random.choice(images.shape[0])
img = plt.figure()
plt.imshow(images[rand_idx].reshape(8, 8), cmap=plt.cm.gray_r)
plt.title(targets[rand_idx])
plt.show()

# --------------- CLASSIFICATION --------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images/255, targets, test_size=0.2)


param_C = 100
param_gamma = 0.5
classifier = svm.SVC(C=param_C, gamma=param_gamma)

print('start classification')
start_time = dt.datetime.now()
classifier.fit(X_train, y_train)
end_time = dt.datetime.now()

print('Trained for {} s'.format(end_time-start_time))

# Plot confusion matric
predicted = classifier.predict(X_test)
cm = metrics.confusion_matrix(y_test, predicted)
plt.figure()
plt.imshow(cm)
plt.colorbar()
plt.tight_layout()
plt.ylabel('True value')
plt.xlabel('Predicted value')
plt.show()

print('Accuracy = {}'.format(metrics.accuracy_score(y_true=y_test, y_pred=predicted)))
