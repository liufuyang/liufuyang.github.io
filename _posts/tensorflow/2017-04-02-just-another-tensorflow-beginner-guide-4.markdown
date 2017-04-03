---
layout: post
title:  "Just another Tensorflow beginner guide (Part4 - Google Cloud)"
date:   2017-04-02
comments: true
---

Now, let's try train our simple sentiment machine learning model on Google cloud. 

## Prepare your Google Cloud Machine Learning Engine
Firstly follow this guide below
https://cloud.google.com/ml-engine/docs/quickstarts/command-line

Basically you need to 
* Go to the cloud console [project page](https://console.cloud.google.com/iam-admin/projects)
* Create a project for this tutorial, let's call it `tf-learn-simple-sentiment`. And you do it by click `Create Project` button:
![gc-prepare-1](/assets/tensorflow/2017-04-02-just-another-tensorflow-beginner-guide-4/gc-prepare-1.png)*Click the CREATE PROJECT button*
* Enable billing for it if you haven't.
* Enable API for this project
![gc-prepare-2](/assets/tensorflow/2017-04-02-just-another-tensorflow-beginner-guide-4/gc-prepare-2.png)*Click the continue button*
* Install the Gooogle Cloud SDK and login. Just follow the step [here](https://cloud.google.com/ml-engine/docs/quickstarts/command-line)
* Verify process by running `gcloud ml-engine models list` and you should see:
`Listed 0 items.d `


```
python -m tensorflow.tensorboard --logdir=./tmp/example-3 --port=8000 --reload_interval=5
```

### Upload training data into Google Cloud storage

Setup bucket and copy the model input data into it:
```
gsutil mb -l europe-west1 gs://tf-learn-simple-sentiment
gsutil cp -r sentiment_set.pickle gs://tf-learn-simple-sentiment/sentiment_set.pickle
```
After this you should be able to see your data file created at google cloud console.
(Click the `Storage` button from the left panel and make sure the project is selected correctly)
![gc-prepare-3](/assets/tensorflow/2017-04-02-just-another-tensorflow-beginner-guide-4/gc-prepare-3.png)*Bucket created*

https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction