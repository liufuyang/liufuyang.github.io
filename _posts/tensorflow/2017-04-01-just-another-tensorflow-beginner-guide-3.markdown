---
layout: post
title:  "Just another Tensorflow beginner guide (Part3 - Keras + GPU)"
date:   2017-04-01
comments: true
---

## MNIST with Keras

You probably have already head about [Keras](https://keras.io/) - a high-level neural networks API, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

If you compare a Keras version of a simple one-layer implementation on the MNIST data, you could feel it's much easier
than the code we shown in [Part 2]({{ site.baseurl }}{% post_url tensorflow/2017-03-17-just-another-tensorflow-beginner-guide-2 %})

```python
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

A more complicated version of a convelutional neural network looks like this:
(or you may want to checkout post [here](https://elitedatascience.com/keras-tutorial-deep-learning-in-python) and [here](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/))
```python
import numpy as np
np.random.seed(123)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

batch_size = 128

# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# 7. Define model architecture
model = Sequential()

model.add(Convolution2D(80, (6, 6), activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(60, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
model.fit(X_train, Y_train,
          batch_size=batch_size, epochs=10, verbose=1,
          validation_data=(X_test, Y_test))
model.save('model.h5')
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

With this implementation the test accuracy can go up to 99.3%.

## Simple sentiment analysis - Keras version

Recall that in [Part 2]({{ site.baseurl }}{% post_url tensorflow/2017-03-17-just-another-tensorflow-beginner-guide-2 %}) we also
tried some sentiment analysis just to show how can we use our own data with 
TensorFlow. 

And here is a code example for trying same but using Keras:
```python
from __future__ import print_function

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import pickle

batch_size = 100
num_classes = 2
epochs = 10

N_X = 423 # len(train_x[0])
layer1_size = 200

# the data, shuffled and split between train and test sets
x_train, y_train, x_test, y_test = pickle.load( open('tmp/sentiment_set.pickle', 'rb' ) )

x_train = x_train.toarray()
x_test = x_test.toarray()
x_train /= np.max(x_train)
x_test /= np.max(x_test)

print(x_train.shape, y_train.shape, 'train samples,', type(x_train[0][0]), ' ', type(y_train[0][0]))
print(x_test.shape,  y_test.shape,  'test samples,',  type(x_test[0][0]),  ' ', type(y_train[0][0]))

# convert class vectors to binary class matrices. Our input already made this. No need to do it again
# y_train = keras.utils.to_categorical(y_train, num_classes) 
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(layer1_size, activation='relu', input_shape=(N_X,)))
model.add(Dropout(0.2))
# Already overfitting, no need to add this extra layer
# model.add(Dense(layer1_size, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test)
                    )
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

This model gives an similar performance as the Tensorflow model showed in [Part 2]({{ site.baseurl }}{% post_url tensorflow/2017-03-17-just-another-tensorflow-beginner-guide-2 %}). 

## About using GPU

To setup a GPU working on your Ubuntu system, you can follow [this guide](https://medium.com/towards-data-science/building-your-own-deep-learning-box-47b918aea1eb). With a GPU doing the calculation, the training speed on GPU for this demo code is **10 times** faster than my Mac 15-inch 
laptop. 

```
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
name: GeForce GTX 980
major: 5 minor: 2 memoryClockRate (GHz) 1.2785
pciBusID 0000:01:00.0
Total memory: 3.94GiB
Free memory: 3.55GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 980, pci bus id: 0000:01:00.0)
60000/60000 [==============================] - 8s - loss: 0.2125 - acc: 0.9352 - val_loss: 0.0402 - val_acc: 0.9857
Epoch 2/15
60000/60000 [==============================] - 7s - loss: 0.0725 - acc: 0.9783 - val_loss: 0.0284 - val_acc: 0.9907
Epoch 3/15
60000/60000 [==============================] - 7s - loss: 0.0534 - acc: 0.9844 - val_loss: 0.0293 - val_acc: 0.9900
...
...
Epoch 12/15
60000/60000 [==============================] - 7s - loss: 0.0165 - acc: 0.9948 - val_loss: 0.0185 - val_acc: 0.9947
Epoch 13/15
60000/60000 [==============================] - 7s - loss: 0.0165 - acc: 0.9947 - val_loss: 0.0259 - val_acc: 0.9928
Epoch 14/15
60000/60000 [==============================] - 7s - loss: 0.0153 - acc: 0.9951 - val_loss: 0.0221 - val_acc: 0.9948
Epoch 15/15
60000/60000 [==============================] - 7s - loss: 0.0156 - acc: 0.9949 - val_loss: 0.0212 - val_acc: 0.9940
[0.021206631667710734, 0.99399999999999999]
```

By running the command `nvidia-smi --loop=1` (--loop=1 means refresh every second)
one can see the CPU usage as below:

```
Sat Apr  1 16:54:59 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.39                 Driver Version: 375.39                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 980     Off  | 0000:01:00.0      On |                  N/A |
| 36%   65C    P2   152W / 195W |   3850MiB /  4035MiB |     91%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      1000    G   /usr/lib/xorg/Xorg                             185MiB |
|    0      3183    G   compiz                                          83MiB |
|    0      4947    G   ...s-passed-by-fd --v8-snapshot-passed-by-fd    57MiB |
|    0      5325    C   python                                        3520MiB |
+-----------------------------------------------------------------------------+
```

However, if you don't have a good GPU besides you, don't worry, we will later have 
a post to use Google Could to do those training calculations. On Google Could 
machine learning platform they provide us [fancy GPUs](https://cloud.google.com/gpu/).