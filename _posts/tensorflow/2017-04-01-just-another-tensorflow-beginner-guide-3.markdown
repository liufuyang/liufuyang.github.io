---
layout: post
title:  "Just another Tensorflow beginner guide (Part3 - Keras + GPU)"
date:   2017-04-01
comments: true
---

You probably have already head about [Keras](https://keras.io/) - a high-level neural networks API, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

If you compare a Keras version of a simple one-layer implementation on the MNIST data, you could feel it's much easier
than the code we shown in [Part 2]({{ site.baseurl }}{% post_url tensorflow/2017-03-17-just-another-tensorflow-beginner-guide-2 %})

```python

```

A more complicated version of a convelutional neural network looks like this:
(or you may want to checkout post [here]() and [here]())
```python
```

To setup a GPU working on your Ubuntu system, you can follow [this guide](https://medium.com/towards-data-science/building-your-own-deep-learning-box-47b918aea1eb)

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
Epoch 4/15
60000/60000 [==============================] - 7s - loss: 0.0437 - acc: 0.9866 - val_loss: 0.0239 - val_acc: 0.9914
Epoch 5/15
60000/60000 [==============================] - 7s - loss: 0.0364 - acc: 0.9896 - val_loss: 0.0220 - val_acc: 0.9932
Epoch 6/15
60000/60000 [==============================] - 7s - loss: 0.0302 - acc: 0.9909 - val_loss: 0.0229 - val_acc: 0.9930
Epoch 7/15
60000/60000 [==============================] - 7s - loss: 0.0278 - acc: 0.9910 - val_loss: 0.0191 - val_acc: 0.9943
Epoch 8/15
60000/60000 [==============================] - 7s - loss: 0.0241 - acc: 0.9928 - val_loss: 0.0204 - val_acc: 0.9934
Epoch 9/15
60000/60000 [==============================] - 7s - loss: 0.0237 - acc: 0.9926 - val_loss: 0.0250 - val_acc: 0.9930
Epoch 10/15
60000/60000 [==============================] - 7s - loss: 0.0219 - acc: 0.9932 - val_loss: 0.0213 - val_acc: 0.9942
Epoch 11/15
60000/60000 [==============================] - 7s - loss: 0.0200 - acc: 0.9942 - val_loss: 0.0191 - val_acc: 0.9951
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
