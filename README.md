# Project 3 - FYS-STK4155
Betina, Ingrid, Polina, Mona and Henrik's venture into Convolutional Neural Networks. Deadline December 14.

### Part 1 (MNIST dataset):
* Betina on SVM (Keras/tensorflow)
* Ingrid+? on CNN implementation (forward + backwords)
* Henrik on CNN (Keras)
```
Choose tasks
```

### Part 2:
* More complex dataset
* Implement SVM
```
If time before deadline
```

Ting som må gjøres:
- prossessere input data slik at softmax virker
- backpropagation i fully connected layer (softmax derivasjon)
- CNN klasse som kobler lagene sammen. Ser for meg en add_layer funksjon, feed_forward og back_propagation

Hvis SVM er klar kan resultatene skrives i rapporten, og teori om SVM også legges til.


We trained a large, deep convolutional neural network to classify the 1.3 million high-resolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 39.7\% and 18.9\% which is considerably better than the previous state-of-the-art results. The neural network, which has 60 million parameters and 500,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and two globally connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of convolutional nets. To reduce overfitting in the globally connected layers we employed a new regularization method that proved to be very effective.
