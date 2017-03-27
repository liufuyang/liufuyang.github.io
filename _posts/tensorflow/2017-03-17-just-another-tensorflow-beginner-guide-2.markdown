---
layout: post
title:  "Just another Tensorflow beginner guide (Part2)"
date:   2017-03-17
comments: true
---

As we have tried some basic Tensorflow code and Tensorboard function in [Part 1]({{ site.baseurl }}{% post_url tensorflow/2017-03-12-just-another-tensorflow-beginner-guide-1 %}), here we are going to try some examples that is 
a bit more complicated

## Study case 1: The MNIST dataset

### Simple model - no hidden layer

You've probably already heard about `MNIST` dataset which contains lots of hand written digits pixel data.
I will not be able to give detailed explanations on how to use tensorflow to do a neural network to classify 
those hand written digits but there are many good resource online. I will link a few below incase you haven't 
read them.

But anyway here is a working example [I took from here](http://ischlag.github.io/2016/06/04/how-to-use-tensorboard/) that use TF to train a simple neural net to classify MNIST dataset, 
which by the end can reach 92% accuracy (without any hidden layer).

```python
# example-2-simple-mnist.py
import tensorflow as tf
from datetime import datetime
import time

# reset everything to rerun in jupyter
tf.reset_default_graph()

# config
batch_size = 100
learning_rate = 0.5
training_epochs = 6
logs_path = './tmp/example-2/' + datetime.now().isoformat()

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/MNIST_data', one_hot=True)

# input images
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input") 
    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

# model parameters will change during training so we use tf.Variable
with tf.name_scope("weights"):
    W = tf.Variable(tf.zeros([784, 10]))

# bias
with tf.name_scope("biases"):
    b = tf.Variable(tf.zeros([10]))

# implement model
with tf.name_scope("softmax"):
    # y is our prediction
    y = tf.nn.softmax(tf.matmul(x,W) + b)

# specify cost function
with tf.name_scope('cross_entropy'):
    # this is our cost
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# specify optimizer
with tf.name_scope('train'):
    # optimizer is an "operation" which we can execute in a session
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# create a summary for our cost and accuracy
train_cost_summary = tf.summary.scalar("train_cost", cross_entropy)
train_acc_summary = tf.summary.scalar("train_accuracy", accuracy)
test_cost_summary = tf.summary.scalar("test_cost", cross_entropy)
test_acc_summary = tf.summary.scalar("test_accuracy", accuracy)

# merge all summaries into a single "operation" which we can execute in a session 
# summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # variables need to be initialized before we can use them
    sess.run(tf.initialize_all_variables())

    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
    # perform training cycles
    for epoch in range(training_epochs):
        
        # number of batches in one epoch
        batch_count = int(mnist.train.num_examples/batch_size)
        
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            # perform the operations we defined earlier on batch
            _, train_cost, train_acc, _train_cost_summary, _train_acc_summary = 
                sess.run([train_op, cross_entropy, accuracy, train_cost_summary, train_acc_summary], 
                    feed_dict={x: batch_x, y_: batch_y})
            # write log
            writer.add_summary(_train_cost_summary, epoch * batch_count + i)
            writer.add_summary(_train_acc_summary, epoch * batch_count + i)

            if i % 100 == 0:
                # for log on test data:
                test_cost, test_acc, _test_cost_summary, _test_acc_summary = 
                    sess.run([cross_entropy, accuracy, test_cost_summary, test_acc_summary], 
                        feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                # write log
                writer.add_summary(_test_cost_summary, epoch * batch_count + i)
                writer.add_summary(_test_acc_summary, epoch * batch_count + i)
                
                print('Epoch {0:3d}, Batch {1:3d} | Train Cost: {2:.2f} | Test Cost: {3:.2f} | Accuracy batch train: {4:.2f} | Accuracy test: {5:.2f}'
                    .format(epoch, i, train_cost, test_cost, train_acc, test_acc))
            
    print('Accuracy: {}'.format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
    print('done')

# tensorboard --logdir=./tmp/example-2 --port=8002 --reload_interval=5
# You can run the following js code in broswer console to make tensooboard to do auto-refresh
# setInterval(function() {document.getElementById('reload-button').click()}, 5000);
```

Now after use python to run this python file, then start up tensooboard with `$ tensooboard --logdir=./tmp/example-2 --port=8002 --reload_interval=5`. Now you can
open in browser at `http://localhost:8002/` then you should be able to see the 
computation graph:

![example1.graph](/assets/tensorflow/2017-03-17-just-another-tensorflow-beginner-guide-2/example1.graph.png)
*Tensorboard graph for simple MNIST model*

Note that in the code we have used `with tf.name_scope('xxx'):` which is used to
group the graph components.

And on the scalar tab you can see some of those scalar summary such as the cost and 
accuracy for training and testing, such as:
![example1.summary](/assets/tensorflow/2017-03-17-just-another-tensorflow-beginner-guide-2/example1.summary.png)*Tensorboard train accuracy summary for simple MNIST model*

---
---
<br><br>
### A bit more complex - Add one hidden layer

As we could see the 92% test accuracy of previous run is not too amazing, 
as we don't have any hidden layer between the input and output vectors. 

Thus we can simply add one hidden layer and see if it will help with increasing 
accuracy in someway. Theoretically it should help as long as the model is not overfitting. 

We can add those changes in the code to add an extra layer:
* Add a parameter somewhere on the top part of code to define the hidden layer size - let's try with 200 
    ```python
    layer1_size = 200
    ```
* Add another weight `W1` and bias `b1` as follow. Note that here we need to initialize 
those weights with `tf.truncated_normal()` method instead of initializing them as zeros like before.
This is because (if I guess correctly) that with hidden layers existing, the feed forward and back propagation
will not be able to contribute any value changes if those weights are initialized as zeros. (You might notice this if you forgot to update the initialization code)
    ```python
    with tf.name_scope("weights"):
        W1 = tf.Variable(tf.truncated_normal([784, layer1_size], stddev=0.1))
        W = tf.Variable(tf.truncated_normal([layer1_size, 10], stddev=1.0))

    with tf.name_scope("biases"):
        b1 = tf.Variable(tf.zeros([layer1_size]))
        b = tf.Variable(tf.zeros([10]))
    ```
* Then add the hidden layer as:
    ```python
    with tf.name_scope('hidden_layers'):
        y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    ```
* Next update the softmax output layer to use hiddlen layer `y1`, also use 
`tf.nn.softmax_cross_entropy_with_logits()` to calculate the cost:
    ```python
    with tf.name_scope("softmax"):
        ylogits = tf.matmul(y1, W) + b
        y = tf.nn.softmax(ylogits)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=ylogits, labels=y_)
        cross_entropy = tf.reduce_mean(cross_entropy)
    ```
* Plus you might want to try a different optimizer:
    ```python
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer().minimize(cross_entropy)
    ```
That should be it, no other change is needed for the rest of the code. And 
if you run this code with one hidden layer (200 nodes), you should get the test
accuracy above 96% :)

For those who want to make it perform even better, I think this [Tensorflow and deep learning without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/index.html) is
a good place to digging the MNIST dataset deeper.

---
---
<br><br>
## Study case 2: A simple sentiment analysis

We can now try a simple sentiment analysis using most of code above but this time 
the fun part is the data will be a bit more realistic and one needs to some simple 
natural language processing on the incoming raw data.

... (to be continued)

