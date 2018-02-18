---
layout: post
title:  "My current deep-learning experiences"
date:   2018-01-02
comments: true
---

I have been learning everything I could about the topic of Machine Learning and Deep Learning since 2016. Mostly via online courese, Youtube videos, books and tutorial projects.

Haven't touched much about Deep Learning during my current work, but I am involved in building and maintaining our Machine Learning components, and we do have some ground level work done to investigate 
some Deep Learning solutions. 

The following are some sources/links to show where I mostly gained my ML/DL knowledge.

## Machine Learning / Deep Learning courses

---

* #### Andrew Ng's Coursera Machine Learning Course
  \[[Course Link](https://www.coursera.org/learn/machine-learning)\] 
	I used a [blog](https://liufuyang.com/) to keep some notes of this courses

---

* #### Geoffrey E. Einton's Coursera Neural Networks Courses:
\[[Course Link](https://www.coursera.org/learn/neural-networks)\] \[[homework code](https://github.com/liufuyang/course-Neural-Networks-for-Machine-Learning)\]

---

* #### A Machine Learning Specialization from University Washington 

	\[[Course Link](https://www.coursera.org/specializations/machine-learning)\] 4 Courses in total. Tought by Professor [Carlos Guestrin
	](https://www.linkedin.com/in/carlos-guestrin-5352a869/) (now Director of Machine Learning at Apple)

	Originally 6 courses planned also covering deeplearning but they
	cut the last two courses after the course provider (together with the company Turi(Dato) supporting it) bought by Apple. A month ago, after two years, Apple finally release the Turi code publicly on [github](https://github.com/apple/turicreate) ... 
	*	Machine Learning Foundations
	* Regression
	* Classification \[[homework code](https://github.com/liufuyang/ML_Coursera_C3_Classification)\] 
		* Logistic Regression, Regularization, Decision Trees, Boosting...
	* Clustering & Retrieval \[[homework code](https://github.com/liufuyang/ML_Coursera_C4_Clustering_Retrieval)\]
		* Nearest Neighbors, Locality Sensitivity Hashing, K-Means-Clustering...
		
---

* #### All the Deep Learning courses so far provided from deeplearning.ai 

	\[[Course Link](https://www.coursera.org/specializations/deep-learning)\] 4 Courses so far
	
	* Neural Networks and Deep Learning \[[homework code](https://github.com/liufuyang/deep_learning_tutorial/tree/master/course-deeplearning.ai/course1-nn-and-deeplearning)\] 
	* Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
	* Structuring Machine Learning Projects
	* Convolutional Neural Networks \[[homework code](https://github.com/liufuyang/deep_learning_tutorial/tree/master/course-deeplearning.ai/course4-cnn)\]

---

* #### A PyTorch course from JiZhi

	\[[Course Link](http://campus.swarma.org/gapp=10346)\] Taught by Zhang Jiang. \[[homework code](https://github.com/liufuyang/PyTorch_homework)\]
	
	Through this course I get some basic knowledge on RNN and Reinforcement Learning

---
---
<br>

## Some workshop/study/hackathon projects

---

* #### Google Codelab - [Tensorflow and deep learning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)

	Some first hands on experiences with Tensorflow and MNIST data. \[[homework code](https://github.com/liufuyang/tensorflow-mnist-tutorial/blob/master/mnist_1.0_convolutional.py)\]

---

* #### A Tradeshift hackathon project to use a simple NN to do sender classification

	To test the idea of using the word count (later on using TF-IDF) to classify the sender of an invoice.

	Things learned:
	
	* One can use a small layer between a big input layer and a big output layer to save number of weights (save memory), and it still works. The information extraction effect of NN is amazing.
	* BatchSize matters a lot when training with GPU. If input features are very big, try use smaller batches to avoid GPU hungry.
		
	\[[A simple Pytorch implementation](https://github.com/liufuyang/ts-hackathon-2017/blob/master/all_data_h2000/train.py)\]
	
	\[[A simple Java Deeplearning4j implementation](https://github.com/liufuyang/ts_sender_classifer/blob/master/src/main/java/SenderClassifierExample.java)\]

---

* #### Tried to replicate the work of OpenAI paper [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444)

	But I didn't go far, only setup a simple runnable LSTM framework, as I realized the dataset is huge and 
	I didn't have a GPU at that moment.

	[Some sample code here](https://github.com/liufuyang/generating-reviews-discovering-sentiment/blob/master/keras-implementation/UtilCode-1%20batch%20generator.ipynb) on using Keras/Tensorflow to do a stateful LSTM to generate characters.

---

* #### MorvanZhou's Pytorch tutorial
	I went through his video tutorial on Pytorch and also helped him making the notebook version of code examples [here](https://github.com/MorvanZhou/PyTorch-Tutorial).

---
---
<br>

## A few blog posts

---

I have very few blog posts which I think helped some readers out there, such as:
* [How to run Keras model on Google Cloud GPU](http://liufuyang.github.io/2017/04/02/just-another-tensorflow-beginner-guide-4.html)

* [How to deploy a Keras model as a api service on Google Cloud](http://liufuyang.github.io/2017/04/07/just-another-tensorflow-beginner-guide-5.html)

I did those Google Cloud examples to prepare a big Kiggle competition, unfortunately I didn't have time  to continue on doing the competition afterwards.

---
---
<br>

## Other toy projects

They are not ML related but I think you might be interested in knowing about.

---

* #### Hosted a small service to gather tweets on Tesla
	\[[Code base here](https://github.com/liufuyang/twitter_tesla)\]. Using Python, Flask and Docker. And it still runs today:
	* http://lifeinweeks.ml:8080/tesla_tweets/count	
	* http://lifeinweeks.ml:8080/tesla_tweets

---

* #### A Vue.js frontend project to show a life-week calendar
	\[[Code base here](https://github.com/liufuyang/lifeinweeks)\]. Frontend with Vue and D3, backend with python (calculating calendar data based on birthday)
	https://lifeinweeks.ml
	
---
---
	