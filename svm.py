# Import libraries
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata

#data set with 28x28 images
mnist = fetch_mldata('MNIST original')
#dict_keys(['DESCR', 'COL_NAMES', 'target', 'data'])
images = mnist.data
targets = mnist.target


"""
#datatset with 8x8 images
digits = datasets.load_digits()
images = digits.images.reshape((len(digits.images), -1))
targets = digits.target
"""
"""
# Show a random image with label
rand_idx = np.random.choice(images.shape[0])
img = plt.figure()
plt.imshow(images[rand_idx].reshape(8, 8), cmap=plt.cm.gray_r)
plt.title(targets[rand_idx])
plt.show()
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images/255, targets, test_size=0.3, train_size = 0.1)#, random_state=42)

"""
# ----- Find best values for c and gamma ------------------
from sklearn.model_selection import GridSearchCV
gamma_values = np.logspace(-3, 0, 4) #array([0.001, 0.01 , 0.1  , 1.   ])
c_values = np.logspace(-1, 1, 3)     #array([ 0.1,  1. , 10. ])
parameters = {'kernel':['rbf'], 'C':c_values, 'gamma': gamma_values}
svm_clsf = svm.SVC()
grid_clsf = GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=1, verbose=2)
grid_clsf.fit(X_train, y_train)
sorted(grid_clsf.cv_results_.keys())
classifier = grid_clsf.best_estimator_
params = grid_clsf.best_params_
scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(c_values),len(gamma_values))
print(scores)
"""

#[[0.70828571 0.901      0.21371429 0.11771429]
# [0.898      0.94514286 0.827      0.11771429]
# [0.92785714 0.95342857 0.83657143 0.11771429]]
#gamma = 0.01
#c = 10

# --------------- CLASSIFICATION --------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images/255, targets, test_size=0.4, train_size = 0.1)#, random_state=42)

param_C = 10
param_gamma = 0.01
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
