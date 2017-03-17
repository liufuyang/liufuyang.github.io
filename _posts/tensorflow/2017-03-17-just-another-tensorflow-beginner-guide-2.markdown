---
layout: post
title:  "Just another Tensorflow beginner guide (Part2)"
date:   2017-03-17
comments: true
---

As we have tried some basic Tensorflow code and Tensorboard function in [Part 1]({{ site.baseurl }}{% post_url tensorflow/2017-03-12-just-another-tensorflow-beginner-guide-1 %}), here we are going to try some examples that is 
a bit more complicated

# Example 1 - The MNIST dataset

You've probably already heard about `MNIST` dataset which contains lots of hand written digits pixel data.
I will not be able to give detailed explanations on how to use tensorflow to do a neural network to classify 
those hand written digits but there are many good resource online. I will link a few below incase you haven't 
read them.

But anyway here is a working example [I took from here](http://ischlag.github.io/2016/06/04/how-to-use-tensorboard/) that use TF to train a simple neural net to classify MNIST dataset, 
which by the end can reach 92% accuracy.

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

# tensooboard --logdir=./tmp/example-2 --port=8002 --reload_interval=5
# You can run the following js code in broswer console to make tensooboard to do auto-refresh
# setInterval(function() {document.getElementById('reload-button').click()}, 5000);
```

Now after use python to run this python file, then start up tensooboard with `$ tensooboard --logdir=./tmp/example-2 --port=8002 --reload_interval=5`. Now you can
open in browser at `http://localhost:8002/` then you should be able to see the 
computation graph and summary metrics such as `cost` and `accuracy`.

![example1.graph](/assets/tensorflow/2017-03-17-just-another-tensorflow-beginner-guide-2/example1.graph.png)*Tensorboard graph for simple MNIST model*
Note that in the code we have used `with tf.name_scope('xxx'):` which is used to
group the graph components.

And on the scalar tab you can see some of those scalar summary such as the cost and 
accuracy for training and testing, such as:
![example1.summary](/assets/tensorflow/2017-03-17-just-another-tensorflow-beginner-guide-2/example1.summary.png)*Tensorboard train accuracy summary for simple MNIST model*
